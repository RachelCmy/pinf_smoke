import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from siren_basic import *
from run_pinf_helpers import BBox_Tool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, dim=3):
    if i == -1:
        return nn.Identity(), dim
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : dim,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, bbox_model=None):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        self.bbox_model = bbox_model

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(input_pts[:,:3])
            outputs = torch.reshape(bbox_mask, [-1,1]) * outputs

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

# Model
class SIREN_NeRFt(nn.Module):
    def __init__(self, D=8, W=256, input_ch=4, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, fading_fin_step=0, bbox_model=None):
        """ 
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super(SIREN_NeRFt, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.fading_step = 0
        self.fading_fin_step = fading_fin_step if fading_fin_step>0 else 0
        self.bbox_model = bbox_model

        first_omega_0 = 30.0
        hidden_omega_0 = 1.0

        self.pts_linears = nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0)] + 
            [SineLayer(W, W, omega_0=hidden_omega_0) 
                if i not in self.skips else SineLayer(W + input_ch, W, omega_0=hidden_omega_0) for i in range(D-1)]
        )
        
        final_alpha_linear = nn.Linear(W, 1)
        self.alpha_linear = final_alpha_linear
      
        if use_viewdirs:
            self.views_linear = SineLayer(input_ch_views, W//2, omega_0=first_omega_0)
            self.feature_linear = SineLayer(W, W//2, omega_0=hidden_omega_0)
            self.feature_view_linears = nn.ModuleList([SineLayer(W, W, omega_0=hidden_omega_0)])
        
        final_rgb_linear = nn.Linear(W, 3)
        self.rgb_linear = final_rgb_linear

    def update_fading_step(self, fading_step):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - radiance_in_step)
        if fading_step >=0:
            self.fading_step = fading_step

    def fading_wei_list(self):
        # try print(fading_wei_list()) for debug
        step_ratio = np.clip(float(self.fading_step)/float(max(1e-8, self.fading_fin_step)), 0, 1)
        ma = 1+(self.D-2)*step_ratio # in range of 1 to self.D-1
        fading_wei_list = [np.clip(1+ma-m,0,1)*np.clip(1+m-ma,0,1) for m in range(self.D)]
        return fading_wei_list

    def print_fading(self):
        w_list = self.fading_wei_list()
        _str = ["h%d:%0.03f"%(i,w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        print("; ".join(_str))

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        h = input_pts
        h_layers = []
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        
        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step: 
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w,y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w*y + h

        alpha = self.alpha_linear(h)

        if self.use_viewdirs:            
            input_pts_feature = self.feature_linear(h)
            input_views_feature = self.views_linear(input_views)

            h = torch.cat([input_pts_feature, input_views_feature], -1)
        
            for i, l in enumerate(self.feature_view_linears):
                h = self.feature_view_linears[i](h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(input_pts[:,:3])
            outputs = torch.reshape(bbox_mask, [-1,1]) * outputs

        return outputs

# Velocity Model
class SIREN_vel(nn.Module):
    def __init__(self, D=6, W=128, input_ch=4, output_ch=3, skips=[], fading_fin_step=0, bbox_model=None):
        """ 
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super(SIREN_vel, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.fading_step = 0
        self.fading_fin_step = fading_fin_step if fading_fin_step>0 else 0
        self.bbox_model = bbox_model

        first_omega_0 = 30.0
        hidden_omega_0 = 1.0

        self.hid_linears = nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0)] + 
            [SineLayer(W, W, omega_0=hidden_omega_0) 
                if i not in self.skips else SineLayer(W + input_ch, W, omega_0=hidden_omega_0) for i in range(D-1)]
        )
        
        final_vel_linear = nn.Linear(W, output_ch)

        self.vel_linear = final_vel_linear

    def update_fading_step(self, fading_step):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - vel_in_step)
        if fading_step >=0:
            self.fading_step = fading_step

    def fading_wei_list(self):
        # try print(fading_wei_list()) for debug
        step_ratio = np.clip(float(self.fading_step)/float(max(1e-8, self.fading_fin_step)), 0, 1)
        ma = 1+(self.D-2)*step_ratio # in range of 1 to self.D-1
        fading_wei_list = [np.clip(1+ma-m,0,1)*np.clip(1+m-ma,0,1) for m in range(self.D)]
        return fading_wei_list

    def print_fading(self):
        w_list = self.fading_wei_list()
        _str = ["h%d:%0.03f"%(i,w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        print("; ".join(_str))

    def forward(self, x):
        h = x
        h_layers = []
        for i, l in enumerate(self.hid_linears):
            h = self.hid_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([x, h], -1)

        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step: 
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w,y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w*y + h
        
        vel_out = self.vel_linear(h)

        if self.bbox_model is not None:
            bbox_mask = self.bbox_model.insideMask(x[...,:3])
            vel_out = torch.reshape(bbox_mask, [-1,1]) * vel_out

        return vel_out


# Model
class SIREN_Hybrid(nn.Module):
    def __init__(self, D=8, W=256, input_ch=4, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, fading_fin_step_static=0, fading_fin_step_dynamic=0, bbox_model=None):
        """ 
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super(SIREN_Hybrid, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.fading_step_static = 0
        self.fading_step_dynamic = 0
        self.fading_fin_step_static = fading_fin_step_static if fading_fin_step_static>0 else 0
        self.fading_fin_step_dynamic = fading_fin_step_dynamic if fading_fin_step_dynamic>0 else 0
        self.bbox_model = bbox_model

        self.static_model = SIREN_NeRFt(D=D, W=W, input_ch=input_ch-1, input_ch_views=input_ch_views, output_ch=output_ch, skips=skips, use_viewdirs=use_viewdirs, fading_fin_step=fading_fin_step_static, bbox_model=bbox_model)

        self.dynamic_model = SIREN_NeRFt(D=D, W=W, input_ch=input_ch, input_ch_views=input_ch_views, output_ch=output_ch, skips=skips, use_viewdirs=use_viewdirs, fading_fin_step=fading_fin_step_dynamic, bbox_model=bbox_model)

    def update_fading_step(self, fading_step_static, fading_step_dynamic):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - static_in_step, global_step - dynamic_in_step)
        self.static_model.update_fading_step(fading_step_static)
        self.dynamic_model.update_fading_step(fading_step_dynamic)

    def fading_wei_list(self, isStatic=False):
        if isStatic:
            return self.static_model.fading_wei_list()
        return self.dynamic_model.fading_wei_list()

    def print_fading(self):
        w_list = self.fading_wei_list(isStatic=True)
        _str = ["static_h%d:%0.03f"%(i,w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        print("; ".join(_str))
        w_list = self.fading_wei_list()
        _str = ["dynamic_h%d:%0.03f"%(i,w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        print("; ".join(_str))

    def forward(self, x):
        inputs_xyz, input_t, input_views = torch.split(x, [self.input_ch-1, 1, self.input_ch_views], dim=-1)
        
        dynamic_x = x
        static_x = torch.cat((inputs_xyz, input_views), dim=-1)

        static_output = self.static_model.forward(static_x)
        dynamic_output = self.dynamic_model.forward(x)
        outputs = torch.cat([static_output, dynamic_output], dim=-1)

        return outputs

    def toDevice(self, device):
        self.static_model = self.static_model.to(device)
        self.dynamic_model = self.dynamic_model.to(device)
        

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    device = weights.get_device()
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
    )  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0.0, 1.0, N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u).to(device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)

    below = torch.max(torch.zeros_like(inds - 1, device=device), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds, device=device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
