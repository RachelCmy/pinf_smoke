import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
    from tqdm import tqdm, trange
else:
    def tqdm(iterable, **kwargs): return iterable
    trange = range

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from nerf_load.load_llff import load_llff_data
from nerf_load.load_deepvoxels import load_dv_data
from nerf_load.load_blender import load_blender_data
from nerf_load.load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
DEBUG = False

def set_rand_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  time_step=None, bkgd_color=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    if time_step != None:
        time_step = time_step.expand(list(rays.shape[0:-1]) + [1])
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction, t)
        rays = torch.cat([rays, time_step], dim=-1)
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    if bkgd_color is not None:
        torch_bkgd_color = torch.Tensor(bkgd_color).to(device)
        # rgb map for model: fine, coarse, merged, dynamic_fine, dynamic_coarse
        for _i in ['_map', '0', 'h1', 'h10', 'h2', 'h20']: #  add background for synthetic scenes, for image-based supervision
            rgb_i, acc_i = 'rgb'+_i, 'acc'+_i
            if (rgb_i in all_ret) and (acc_i in all_ret):
                all_ret[rgb_i] = all_ret[rgb_i] + torch_bkgd_color*(1.-all_ret[acc_i][..., None])

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, render_steps=None, bkgd_color=None):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    cur_timestep = None
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        if render_steps is not None:
            cur_timestep = render_steps[i]
        t = time.time()
        rgb, disp, acc, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], time_step=cur_timestep, bkgd_color=bkgd_color, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            other_rgbs = []
            if gt_imgs is not None:
                other_rgbs.append(gt_imgs[i])
            for rgb_i in ['rgbh1','rgbh2','rgb0']: 
                if rgb_i in extras:
                    _data = extras[rgb_i].cpu().numpy()
                    other_rgbs.append(_data)
            if len(other_rgbs) >= 1:
                other_rgb8 = np.concatenate(other_rgbs, axis=1)
                other_rgb8 = to8b(other_rgb8)
                filename = os.path.join(savedir, '_{:03d}.png'.format(i))
                imageio.imwrite(filename, other_rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args, vel_model=None, bbox_model=None, ndim=3):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, ndim)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed, dim=ndim)
    output_ch = 4 # 5 if args.N_importance > 0 else 4
    skips = [4]

    my_model_dict = {
        "nerf":NeRF,
        "siren":SIREN_NeRFt,
        "hybrid":SIREN_Hybrid,
    }
    model_args = {}
    if args.fading_layers > 0:
        if args.net_model == "siren":
            model_args["fading_fin_step"] = args.fading_layers
        elif args.net_model == "hybrid":
            model_args["fading_fin_step_static"] = args.fading_layers
            model_args["fading_fin_step_dynamic"] = args.fading_layers
    if bbox_model is not None:
        model_args["bbox_model"] = bbox_model

    my_model = my_model_dict[args.net_model]

    model = my_model(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, **model_args)
    if args.net_model == "hybrid":
        model.toDevice(device)
    model = model.to(device)
    
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = my_model(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, **model_args)
        if args.net_model == "hybrid":
            model_fine.toDevice(device)
        model_fine = model_fine.to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    vel_optimizer = None
    if vel_model is not None:
        vel_grad_vars = list(vel_model.parameters())
        vel_optimizer = torch.optim.Adam(params=vel_grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################
    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # Load model
        if args.net_model == "hybrid":
            model.static_model.load_state_dict(ckpt['network_fn_state_dict_static'])
            if model_fine is not None:
                model_fine.static_model.load_state_dict(ckpt['network_fine_state_dict_static'])
            model.dynamic_model.load_state_dict(ckpt['network_fn_state_dict_dynamic'])
            if model_fine is not None:
                model_fine.dynamic_model.load_state_dict(ckpt['network_fine_state_dict_dynamic'])
        else:
            model.load_state_dict(ckpt['network_fn_state_dict'])
            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        
        if vel_model is not None:
            if 'network_vel_state_dict' in ckpt:
                vel_model.load_state_dict(ckpt['network_vel_state_dict'])
        if vel_optimizer is not None:
            if 'vel_optimizer_state_dict' in ckpt:
                vel_optimizer.load_state_dict(ckpt['vel_optimizer_state_dict'])
    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, vel_optimizer


def raw2outputs(raw_list, z_vals, rays_d, raw_noise_std=0, pytest=False, remove99=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw_list: a list of tensors in shape [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    noise = 0.
    alpha_list = []
    color_list = []
    for raw in raw_list:
        if raw is None: continue
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(42)
                noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)
        
        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        if remove99:
            alpha = torch.where(alpha > 0.99, torch.zeros_like(alpha), alpha)
        rgb = torch.sigmoid(raw[..., :3]) # [N_rays, N_samples, 3]

        alpha_list += [alpha]
        color_list += [rgb]
    
    densTiStack = torch.stack([1.-alpha for alpha in alpha_list], dim=-1) 
    # [N_rays, N_samples, N_raws]
    densTi = torch.prod(densTiStack, dim=-1, keepdim=True) 
    # [N_rays, N_samples]
    densTi_all = torch.cat([densTiStack, densTi], dim=-1) 
    # [N_rays, N_samples, N_raws + 1] 
    Ti_all = torch.cumprod(densTi_all + 1e-10, dim=-2) # accu along samples
    Ti_all = Ti_all / (densTi_all + 1e-10)
    # [N_rays, N_samples, N_raws + 1], exclusive
    weights_list = [alpha * Ti_all[...,-1] for alpha in alpha_list] # a list of [N_rays, N_samples]
    self_weights_list = [alpha_list[alpha_i] * Ti_all[...,alpha_i] for alpha_i in range(len(alpha_list))] # a list of [N_rays, N_samples]

    def weighted_sum_of_samples(wei_list, content_list=None, content=None):
        content_map_list = []
        if content_list is not None:
            content_map_list = [
                torch.sum(weights[..., None] * ct, dim=-2)  
                # [N_rays, N_content], weighted sum along samples
                for weights, ct in zip(wei_list, content_list)
            ]
        elif content is not None:
            content_map_list = [
                torch.sum(weights * content, dim=-1)  
                # [N_rays], weighted sum along samples
                for weights in wei_list
            ]
        content_map = torch.stack(content_map_list, dim=-1) 
        # [N_rays, (N_contentlist,) N_raws]
        content_sum = torch.sum(content_map, dim=-1) 
        # [N_rays, (N_contentlist,)]
        return content_sum, content_map

    rgb_map, _ = weighted_sum_of_samples(weights_list, color_list) # [N_rays, 3]
    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map, _ = weighted_sum_of_samples(weights_list, None, 1) # [N_rays]

    _, rgb_map_stack = weighted_sum_of_samples(self_weights_list, color_list)
    _, acc_map_stack = weighted_sum_of_samples(self_weights_list, None, 1)

    # Estimated depth map is expected distance.
    # Disparity map is inverse depth.
    depth_map,_ = weighted_sum_of_samples(weights_list, None, z_vals) # [N_rays]
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)
    # alpha * Ti
    weights = (1.-densTi)[...,0] * Ti_all[...,-1] # [N_rays, N_samples]
    
    # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    # depth_map = torch.sum(weights * z_vals, -1)
    # acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map, Ti_all[...,-1], rgb_map_stack, acc_map_stack


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                has_t = False,
                vel_model=None,
                netchunk=1024*64,
                warp_fading_dt=None,
                warp_mod="rand",
                remove99=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

      warp_fading_dt, to train nearby frames with flow-based warping, fading*delt_t
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    rays_t, viewdirs = None, None
    if has_t:
        rays_t = ray_batch[:,-1:] # [N_rays, 1]
        viewdirs = ray_batch[:, -4:-1] if ray_batch.shape[-1] > 9 else None
    elif ray_batch.shape[-1] > 8:
        viewdirs = ray_batch[:,-3:]

    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(42)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    if rays_t is not None:
        rays_t_bc = torch.reshape(rays_t, [-1,1,1]).expand([N_rays, N_samples, 1])
        pts = torch.cat([pts, rays_t_bc], dim = -1)

    def warp_raw_random(orig_pts, orig_dir, fading, fn, mod="rand", has_t=has_t):
        # mod, "rand", "forw", "back", "none"
        if (not has_t) or (mod=="none") or (vel_model is None):
            orig_raw = network_query_fn(orig_pts, orig_dir, fn)  # [N_rays, N_samples, 4]
            return orig_raw

        orig_pos, orig_t = torch.split(orig_pts, [3, 1], -1)
        
        _vel = batchify(vel_model, netchunk)(orig_pts.view(-1,4))
        _vel = torch.reshape(_vel, [N_rays, -1, 3])
        # _vel.shape, [N_rays, N_samples(+N_importance), 3]
        if mod=="rand":
            # random_warpT = np.random.normal(0.0, 0.6, orig_t.get_shape().as_list())
            # random_warpT = np.random.uniform(-3.0, 3.0, orig_t.shape)
            random_warpT = torch.rand(orig_t.shape)*6.0 -3.0 # [-3,3]
        else:
            random_warpT = 1.0 if mod == "back" else (-1.0) # back
        # mean and standard deviation: 0.0, 0.6, so that 3sigma < 2, train +/- 2*delta_T
        random_warpT = random_warpT * fading
        random_warpT = torch.Tensor(random_warpT)

        warp_t = orig_t + random_warpT
        warp_pos = orig_pos + _vel * random_warpT
        warp_pts = torch.cat([warp_pos, warp_t], dim = -1)
        warp_pts = warp_pts.detach() # stop gradiant

        warped_raw = network_query_fn(warp_pts, orig_dir, fn)  # [N_rays, N_samples, 4]

        return warped_raw

    def get_raw(fn, staticpts, staticdirs, has_t=has_t):
        static_raw, smoke_raw = None, None
        smoke_warp_mod = warp_mod
        if (None in [vel_model, warp_fading_dt]) or (not has_t):
            smoke_warp_mod = "none"
            
        smoke_raw = warp_raw_random(staticpts, staticdirs, warp_fading_dt, fn, mod=smoke_warp_mod, has_t=has_t)
        if has_t and (smoke_raw.shape[-1] > 4): # hybrid mode
            if smoke_warp_mod == "none":
                static_raw = smoke_raw
            else:
                static_raw = warp_raw_random(staticpts, staticdirs, warp_fading_dt, fn, mod="none", has_t=True)

            static_raw = static_raw[..., :4]
            smoke_raw = smoke_raw[..., -4:]
                
        return smoke_raw, static_raw # [N_rays, N_samples, 4], [N_rays, N_samples, 4]

    # raw = run_network(pts)
    C_smokeRaw, C_staticRaw = get_raw(network_fn, pts, viewdirs)
    raw = [C_smokeRaw, C_staticRaw]
    rgb_map, disp_map, acc_map, weights, depth_map, ti_map, rgb_map_stack, acc_map_stack = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest, remove99=remove99)

    if raw[-1] is not None:
        rgbh2_map = rgb_map_stack[...,0] # dynamic
        acch2_map = acc_map_stack[...,0] # dynamic
        rgbh1_map = rgb_map_stack[...,1] # staitc
        acch1_map = acc_map_stack[...,1] # staitc
    
    # raw = network_query_fn(pts, viewdirs, network_fn)
    # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        if rays_t is not None:
            rays_t_bc = torch.reshape(rays_t, [-1,1,1]).expand([N_rays, N_samples+N_importance, 1])
            pts = torch.cat([pts, rays_t_bc], dim = -1)
        
        run_fn = network_fn if network_fine is None else network_fine
        F_smokeRaw, F_staticRaw = get_raw(run_fn, pts, viewdirs)
        raw = [F_smokeRaw, F_staticRaw]

        rgb_map, disp_map, acc_map, weights, depth_map, ti_map, rgb_map_stack, acc_map_stack = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest, remove99=remove99)

        if raw[-1] is not None:
            rgbh20_map = rgbh2_map
            acch20_map = acch2_map
            rgbh10_map = rgbh1_map
            acch10_map = acch1_map
            rgbh2_map = rgb_map_stack[...,0]
            acch2_map = acc_map_stack[...,0]
            rgbh1_map = rgb_map_stack[...,1]
            acch1_map = acc_map_stack[...,1]
        
        # raw = run_network(pts, fn=run_fn)
        # raw = network_query_fn(pts, viewdirs, run_fn)
        # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw[0]
        if raw[1] is not None:
            ret['raw_static'] = raw[1]
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
    
    if raw[-1] is not None:
        ret['rgbh1'] = rgbh1_map
        ret['acch1'] = acch1_map
        ret['rgbh2'] = rgbh2_map
        ret['acch2'] = acch2_map
        if N_importance > 0:
            ret['rgbh10'] = rgbh10_map
            ret['acch10'] = acch10_map
            ret['rgbh20'] = rgbh20_map
            ret['acch20'] = acch20_map
        ret['rgbM'] = rgbh1_map * 0.5 + rgbh2_map * 0.5

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--net_model", type=str, default='nerf',
                        help='which model to use, nerf, siren, hybrid..')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--fix_seed", type=int, default=42,
                        help='the random seed.')
    parser.add_argument("--fading_layers", type=int, default=-1,
                        help='for siren and hybrid models, the step to finish fading model layers one by one during training.')
    parser.add_argument("--tempo_delay", type=int, default=0,
                        help='for hybrid models, the step to start learning the temporal dynamic component.')
    parser.add_argument("--vel_delay", type=int, default=10000,
                        help='for siren and hybrid models, the step to start learning the velocity.')
    parser.add_argument("--N_iter", type=int, default=200000,
                        help='for siren and hybrid models, the step to start learning the velocity.')  
    parser.add_argument("--train_warp", default=False, action='store_true',
                        help='train radiance model with velocity warpping')

    # model options
    parser.add_argument("--bbox_min", type=str,
                        default='', help='use a boundingbox, the minXYZ')
    parser.add_argument("--bbox_max", type=str,
                        default='1.0,1.0,1.0', help='use a boundingbox, the maxXYZ')

    # loss hyper params, negative values means to disable the loss terms
    parser.add_argument("--vgg_strides", type=int, default=4,
                        help='vgg stride, should >= 2')
    parser.add_argument("--ghostW", type=float,
                        default=-0.0, help='weight for the ghost density regularization')
    parser.add_argument("--vggW", type=float,
                        default=-0.0, help='weight for the VGG loss')
    parser.add_argument("--overlayW", type=float,
                        default=-0.0, help='weight for the overlay regularization')
    parser.add_argument("--d2vW", type=float,
                        default=-0.0, help='weight for the d2v loss')
    parser.add_argument("--nseW", type=float,
                        default=0.001, help='velocity model, training weight for the physical equations')
    
    # task params
    parser.add_argument("--vol_output_only", action='store_true', 
                        help='do not optimize, reload weights and output volumetric density and velocity')
    parser.add_argument("--vol_output_W", type=int, default=256, 
                        help='In output mode: the output resolution along x; In training mode: the sampling resolution for training')
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a given bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", type=str, default='normal', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=400, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=2000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=25000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
                        
    return parser


def train(parser, args):

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res in ["True", "half"], args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd is not None:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])*args.white_bkgd
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res in ["True", "half"], args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd is not None:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])*args.white_bkgd
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, vel_optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor, bkgd_color=args.white_bkgd)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = args.N_iter + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=False,
                                                bkgd_color=args.white_bkgd,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        # trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, bkgd_color=args.white_bkgd)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir, bkgd_color=args.white_bkgd)
            print('Saved test set')


    
        if i%args.i_print==0:
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            sys.stdout.flush()

        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = config_parser()
    args = parser.parse_args()
    set_rand_seed(args.fix_seed)

    bkg_flag = args.white_bkgd
    args.white_bkgd = np.ones([3], dtype=np.float32) if bkg_flag else None

    if args.dataset_type != 'pinf_data':
        train(parser, args)
    else:
        print("Try 'python run_pinf.py' with the config file instead.")
