import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import requests
import os.path
from run_pinf_helpers import curl2D, curl3D

""" Import pre-trained D2V model as a fixed PyTorch Model
D2V model: https://github.com/RachelCmy/den2vel a tensorflow model translating density to velocity
"""

def download_file(filename, url):
    """
    Download an URL to a file
    """
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)

class MyFCLayer(nn.Module):
    def __init__(self, fc_weight, act):
        super(MyFCLayer, self).__init__()
        self.wei = torch.Tensor(fc_weight)
        self.act = act

    def forward(self, input):
        out = torch.matmul(input, self.wei)
        if self.act is not None and self.act == 'relu':
            out = F.relu(out)
        return out

class MyNormLayer(nn.Module):
    def __init__(self, is2D, shift, scale):
        super(MyNormLayer, self).__init__()
        self.is2D = is2D
        tar_shape = [1,-1,1,1]
        if not is2D:
            tar_shape.append(1)
        self.shift = torch.Tensor(np.reshape(shift,tar_shape))
        self.scale = torch.Tensor(np.reshape(scale,tar_shape))

    def forward(self, input):
        epsilon = 1e-5
        axis_ = [2,3]
        if not self.is2D: axis_ +=[4]
        # input shape, Batch,channel,H,W,D

        sigma, mean = torch.var_mean(input, axis_, unbiased=False, keepdim=True)
        normalized = (input - mean) / torch.sqrt(sigma + epsilon) 
        out = self.scale * normalized + self.shift

        return out


def _padding_size_2d(in_height, in_width, filter_size, stride):
    if in_height % stride == 0:
        pad_along_height = max(filter_size - stride, 0)
    else:
        pad_along_height = max(filter_size - (in_height % stride), 0)
    if in_width % stride == 0:
        pad_along_width = max(filter_size - stride, 0)
    else:
        pad_along_width = max(filter_size - (in_width % stride), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom

def _padding_size_3d(in_depth, in_height, in_width, filter_size, stride):
    if in_depth % stride == 0:
        pad_along_depth = max(filter_size - stride, 0)
    else:
        pad_along_depth = max(filter_size - (in_depth % stride), 0)
    if in_height % stride == 0:
        pad_along_height = max(filter_size - stride, 0)
    else:
        pad_along_height = max(filter_size - (in_height % stride), 0)
    if in_width % stride == 0:
        pad_along_width = max(filter_size - stride, 0)
    else:
        pad_along_width = max(filter_size - (in_width % stride), 0)
    pad_front = pad_along_depth // 2
    pad_back = pad_along_depth - pad_front
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    
    return pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back
    
def _my_pad_fn(input, padding, mode, is2D, name):
    ori_shape = list(input.shape)
    if mode == 'same' or is2D:
        out = F.pad(input, padding, 'constant')
    else:
        input2D = torch.reshape(input, [ori_shape[0], ori_shape[1]*ori_shape[2], ori_shape[3], ori_shape[4]])
        # b, c*d, h,w
        pad_2D = F.pad(input2D, (padding[0],padding[1],padding[2],padding[3]), 'reflect')
        pad_2D = pad_2D.view(ori_shape[0], ori_shape[1], ori_shape[2], pad_2D.shape[-2], pad_2D.shape[-1])
        
        pad_flip = torch.flip(pad_2D, (2,))
        pad_front = pad_flip[:,:,-padding[4]-1:-1,...] # pad_2D[:,:, padding[4]:0:-1, ...]
        pad_back  = pad_flip[:,:,1:1+padding[5],...] # pad_2D[:,:, -2:-2-padding[5]:-1, ...]
        out = torch.cat([pad_front, pad_2D, pad_back],dim=2)

    # assert out.shape[0] == ori_shape[0]
    # assert out.shape[1] == ori_shape[1]
    # assert out.shape[2] == ori_shape[2] + padding[4] + padding[5]
    # assert out.shape[3] == ori_shape[3] + padding[2] + padding[3]
    # assert out.shape[4] == ori_shape[4] + padding[0] + padding[1]
    return out
    
    
class FixLayer(object):
    # base class for Generator and Discriminator    
    def __init__(self, _var_list, _var_name):
        self.var_list = _var_list
        self.var_name = _var_name
        self.var_idx = 0
        self._print = False 
        # set to True to check if d2v variables are correctly loaded

    def get_variable(self, name, shape):
        test_var = self.var_list[self.var_idx]
        if self._print:
            print("[loading d2v variables]", name, shape, "get var %d" % self.var_idx, self.var_name[self.var_idx], test_var.shape)
        self.var_idx += 1
        return test_var

    def _norm(self, name, is2D, channel):
        # input shape, Batch,channel,H,W,D
        shift = self.get_variable(name+'.norm_shift', shape=[channel])
        scale = self.get_variable(name+'.norm_scale', shape=[channel])
        return MyNormLayer(is2D, shift, scale)

    def _conv(self, ch_in, num_filters, name, k_size, stride, is2D=True, hasbias=True):
        conv_fn = nn.Conv2d if is2D else nn.Conv3d
        filter_shape = [num_filters, ch_in, k_size, k_size]
        bias_shape = [num_filters]
        transpose_order = [-1, -2, 0, 1]
        if not is2D:
            filter_shape.append(k_size)
            transpose_order.append(2)

        cur_layer = conv_fn(ch_in, num_filters, k_size, stride)
        with torch.no_grad():
            np_wei = self.get_variable(name+'.conv_w', filter_shape)
            # kernel_size * kernel_size * kernel_size *  ch_out * ch_in
            # [ch_out, ch_in, kernel_size, kernel_size, kernel_size]
            np_wei = np_wei.transpose(transpose_order)
            cur_layer.weight = torch.nn.Parameter(torch.Tensor(np_wei))
            if hasbias:
                np_bias = self.get_variable(name+'.conv_b', bias_shape)
                np_bias = np.reshape(np_bias, bias_shape)
            else:
                np_bias = np.zeros(bias_shape)
            cur_layer.bias = torch.nn.Parameter(torch.Tensor(np_bias))
        cur_layer.requires_grad_(False)

        return cur_layer

    def _deconv(self, ch_in, num_filters, name, k_size, stride, is2D=True, hasbias=True):
        deconv_fn = nn.ConvTranspose2d if is2D else nn.ConvTranspose3d
        filter_shape = [ch_in, num_filters, k_size, k_size]
        bias_shape = [num_filters]
        transpose_order = [-1, -2, 0, 1]
        if not is2D:
            filter_shape.append(k_size)
            transpose_order.append(2)
        cur_layer = deconv_fn(ch_in, num_filters, k_size, stride)
        with torch.no_grad():
            np_wei = self.get_variable(name+'.deconv_w', filter_shape)
            # kernel_size * kernel_size * kernel_size *  ch_out * ch_in
            # [ch_out, ch_in, kernel_size, kernel_size, kernel_size]
            np_wei = np_wei.transpose(transpose_order)
            cur_layer.weight = torch.nn.Parameter(torch.Tensor(np_wei))
            if hasbias:
                np_bias = self.get_variable(name+'.deconv_b', bias_shape)
                np_bias = np.reshape(np_bias, bias_shape)
            else:
                np_bias = np.zeros(bias_shape)
            cur_layer.bias = torch.nn.Parameter(torch.Tensor(np_bias))
        cur_layer.requires_grad_(False)

        return cur_layer
        
    def _fully_connected(self, ch_in, out_n, act, name):
        fc_shape = [ch_in, out_n]
        np_wei = self.get_variable(name+'.fc_w', fc_shape)
        # cur_layer = nn.Linear(ch_in, out_n, bias=False)
        return MyFCLayer(np_wei, act)

    def _residual(self, num_filters, name, is2D=True, norm='instance'):
        layer_norm1,layer_norm2 = None, None
        layer_res1 = self._conv(num_filters,num_filters,name+'.res1',3,1,is2D,hasbias=False)
        if norm is not None:
            if norm == 'instance':
                layer_norm1 = self._norm(name+'.norm1', is2D, num_filters)
        layer_res2 = self._conv(num_filters,num_filters,name+'.res2',3,1,is2D,hasbias=False)
        if norm is not None:
            if norm == 'instance':
                layer_norm2 = self._norm(name+'.norm2', is2D, num_filters)
        return layer_res1, layer_res2, layer_norm1, layer_norm2 

class MyConvBlock(nn.Module):
    def __init__(self, fix_layer_tool, ch_in, num_filters, name, k_size, stride, 
        pad='reflect',is2D=True, norm='instance',activation='relu'):
        assert norm in ['instance', None]
        assert activation in ['relu', None]
        super(MyConvBlock, self).__init__()
        self.conv_layer = fix_layer_tool._conv(ch_in, num_filters, name+'.CB', 
            k_size, stride,is2D)
        self.norm_layer = None
        if norm is not None:
            if norm == 'instance':
                self.norm_layer = fix_layer_tool._norm(name+'.CB', is2D, num_filters)
        self.act = activation
        self.is2D = is2D
        self.k_size = k_size
        self.stride = stride
        self.pad = pad
        self.name = name

    def forward(self, input):
        if self.pad in ['reflect', 'same']:
            if self.is2D:
                my_pad = _padding_size_2d(input.shape[-2],input.shape[-1],self.k_size,self.stride)
            else:
                my_pad = _padding_size_3d(input.shape[-3],input.shape[-2],input.shape[-1],self.k_size,self.stride)
            input = _my_pad_fn(input, my_pad, self.pad, self.is2D, self.name)
        out = self.conv_layer(input)
        if self.norm_layer is not None:
            out = self.norm_layer(out)
        if self.act is not None:
            if self.act == 'relu':
                out = F.relu(out)
        return out

class MyDeconvBlock(nn.Module):
    def __init__(self, fix_layer_tool, ch_in, num_filters, name, k_size, stride, 
        is2D=True, norm='instance',activation='relu'):
        assert norm in ['instance', None]
        assert activation in ['relu', None]
        super(MyDeconvBlock, self).__init__()
        self.deconv_layer = fix_layer_tool._deconv(ch_in, num_filters, name+'.DCB', 
            k_size, stride,is2D)
        self.norm_layer = None
        if norm is not None:
            if norm == 'instance':
                self.norm_layer = fix_layer_tool._norm(name+'.DCB', is2D, num_filters)
        self.act = activation
        self.is2D = is2D
        self.k_size = k_size
        self.stride = stride

    def forward(self, input):
        out = self.deconv_layer(input)
        if self.is2D:
            has_pad = _padding_size_2d(input.shape[-2]*self.stride,input.shape[-1]*self.stride,self.k_size,self.stride)
            out = out[:,:,has_pad[2]:out.shape[-2]-has_pad[3],has_pad[0]:out.shape[-1]-has_pad[1]]
        else:
            has_pad = _padding_size_3d(input.shape[-3]*self.stride,input.shape[-2]*self.stride,input.shape[-1]*self.stride,self.k_size,self.stride)
            out = out[:,:,has_pad[4]:out.shape[-3]-has_pad[5],has_pad[2]:out.shape[-2]-has_pad[3],has_pad[0]:out.shape[-1]-has_pad[1]]
        
        if self.norm_layer is not None:
            out = self.norm_layer(out)
        if self.act is not None:
            if self.act == 'relu':
                out = F.relu(out)
        return out

class MyResBlock(nn.Module):
    def __init__(self, fix_layer_tool, num_filters, name, 
        pad='reflect',is2D=True,norm='instance'):
        assert norm in ['instance', None]
        super(MyResBlock, self).__init__()
        self.res_layers = fix_layer_tool._residual(num_filters, name+'.RES',is2D,norm=norm)
        self.is2D = is2D
        self.pad = pad
        self.name = name

    def forward(self, input):
        out = input
        if self.pad in ['reflect', 'same']:
            if self.is2D:
                my_pad = _padding_size_2d(input.shape[-2],input.shape[-1],3,1)
            else:
                my_pad = _padding_size_3d(input.shape[-3],input.shape[-2],input.shape[-1],3,1)
            out = _my_pad_fn(input, my_pad, self.pad, self.is2D, self.name)
        
        out = self.res_layers[0](out)
        if self.res_layers[2] is not None:
            out = self.res_layers[2](out)
        out = F.relu(out)
        
        if self.pad in ['reflect', 'same']:
            if self.is2D:
                my_pad = _padding_size_2d(out.shape[-2],out.shape[-1],3,1)
            else:
                my_pad = _padding_size_3d(out.shape[-3],out.shape[-2],out.shape[-1],3,1)
            out = _my_pad_fn(out, my_pad, self.pad, self.is2D, self.name)
        out = self.res_layers[1](out)
        if self.res_layers[3] is not None:
            out = self.res_layers[3](out)

        return F.relu(input + out) # residuals


class DenToVel(nn.Module):
    def init_configuration(self, obsFlag=False,withZoom=True):
        # set all hyper-parameters for den2vel model
        updateDict = {
            'OpenBounds':True,
            'adv_order':2,
            'buoy': 2.0, 
            'is2D':False,
            'useEnergy':True,
            'usePhy':True, 
            'encPhy':True,
            'useVortEnd':True,
            'obsFlags':obsFlag,
            'obsMoving':obsFlag,
            'batch_size':1, # 1 step
            'crop_size':64, # 3D
            'mid_ch':16,
            'phy_len':2,
            'num_resblock':6,
            'zoom_factor':4.0 if withZoom else -1.0,
            'blend_st':-1,
            'Dst_Flag':0,
            'selfPhy': False,
            'withRef': False, 
            'mode':'inference'
        }
        namelist = list(updateDict)
        valuelist = list(map(updateDict.get, namelist))
        
        Params = collections.namedtuple('Params', ",".join(namelist))
        tmpFLAGS = Params._make(valuelist)
        #print(tmpFLAGS)
        return tmpFLAGS

    def encode_param_layer(self, FLAGS, ch_in):
        # mid_in FLAGS.batch_size x m, 8x8x16
        name = 'encPhy.'
        mid_ch = FLAGS.mid_ch
        dim = 2 if FLAGS.is2D else 3
        mid_shape = [mid_ch]+[8]*dim

        M3 = MyConvBlock(self.fix_layer_tool, ch_in, mid_ch, name+'d16c7', 7, 1, 
            pad='reflect',is2D=FLAGS.is2D, norm=None, activation=self._activation)
                
        if FLAGS.useEnergy: 
            E2 = MyConvBlock(self.fix_layer_tool, mid_ch, mid_ch//2, name+'d8', 3, 2, 
                pad='same',is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
            
            E1 = MyConvBlock(self.fix_layer_tool, mid_ch//2, mid_ch//4, name+'d4', 3, 2, 
                pad='same',is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
                                
            tar_depth = 1
            E0 = MyConvBlock(self.fix_layer_tool, mid_ch//4, tar_depth, name+'d3s1-2', 3, 1, 
                pad='same',is2D=FLAGS.is2D, norm=None, activation=None)
            
            self.KE_list = [E2,E1,E0]
        else:
            self.KE_list = []
                    
        M1 = MyConvBlock(self.fix_layer_tool, mid_ch, mid_ch, name+'d16c3', 3, 1, 
            pad='same',is2D=FLAGS.is2D, norm=None, activation=None)
        M0 = self.fix_layer_tool._fully_connected(np.prod(mid_shape), FLAGS.phy_len, act=None, name=name+'m1fc2')
        self.PHY_list = [M3, M1, M0]
    
    def build_param_layer(self, FLAGS):
        param_n = FLAGS.phy_len
        dim = 2 if FLAGS.is2D else 3
        mid_ch = FLAGS.mid_ch
        mid_shape = [mid_ch]+[8]*dim
        name = 'buildPhy.'

        M1 = self.fix_layer_tool._fully_connected(param_n * 2, np.prod(mid_shape), 'relu',
                                    name+'m1fc16')
        M2 = MyConvBlock(self.fix_layer_tool, mid_ch, mid_ch, name+'d16', 7, 1, 
                pad='reflect',is2D=FLAGS.is2D, norm=None, activation=None)
            
        if FLAGS.useEnergy:      
            energy_c1 = MyConvBlock(self.fix_layer_tool, 2, 2, name+'energy-d2', 3, 1, 
                pad='reflect',is2D=FLAGS.is2D, norm=None, activation=None)
            energy_conv = MyConvBlock(self.fix_layer_tool, 2, 1, name+'energy-d1', 3, 1, 
                pad='reflect',is2D=FLAGS.is2D, norm=None, activation=None)
        self.PHYin_list = [M1, M2]
        self.KEin_list = [energy_c1, energy_conv]
        

    def __init__(self, ckpt='./data/d2v_3Dmodel.npz', obsFlag=False,withZoom=False):
        super(DenToVel,self).__init__()
        FLAGS = self.init_configuration( obsFlag, withZoom)
        # 110 variables, load all weights as constant/fixed variable
        namelist = [
            "generator/zoom_in/LS-8/w", "generator/zoom_in/LS-8/b", "generator/zoom_in/LS-out/w", "generator/zoom_in/LS-out/b", "generator/c7s1-32/w", "generator/c7s1-32/b", "generator/c7s1-32/instance_norm/shift", "generator/c7s1-32/instance_norm/scale", "generator/d64/w", "generator/d64/b", "generator/d64/instance_norm/shift", "generator/d64/instance_norm/scale", "generator/d128/w", "generator/d128/b", "generator/d128/instance_norm/shift", "generator/d128/instance_norm/scale", "generator/R128_0/res1/w", "generator/R128_0/res1/instance_norm/shift", "generator/R128_0/res1/instance_norm/scale", "generator/R128_0/res2/w", "generator/R128_0/res2/instance_norm/shift", "generator/R128_0/res2/instance_norm/scale", "generator/R128_1/res1/w", "generator/R128_1/res1/instance_norm/shift", "generator/R128_1/res1/instance_norm/scale", "generator/R128_1/res2/w", "generator/R128_1/res2/instance_norm/shift", "generator/R128_1/res2/instance_norm/scale", "generator/R128_2/res1/w", "generator/R128_2/res1/instance_norm/shift", "generator/R128_2/res1/instance_norm/scale", "generator/R128_2/res2/w", "generator/R128_2/res2/instance_norm/shift", "generator/R128_2/res2/instance_norm/scale", "generator/R128_3/res1/w", "generator/R128_3/res1/instance_norm/shift", "generator/R128_3/res1/instance_norm/scale", "generator/R128_3/res2/w", "generator/R128_3/res2/instance_norm/shift", "generator/R128_3/res2/instance_norm/scale", "generator/encPhy/d16c7/w", "generator/encPhy/d16c7/b", "generator/encPhy/d8/w", "generator/encPhy/d8/b", "generator/encPhy/d8/instance_norm/shift", "generator/encPhy/d8/instance_norm/scale", "generator/encPhy/d4/w", "generator/encPhy/d4/b", "generator/encPhy/d4/instance_norm/shift", "generator/encPhy/d4/instance_norm/scale", "generator/encPhy/d3s1-2/w", "generator/encPhy/d3s1-2/b", "generator/encPhy/d16c3/w", "generator/encPhy/d16c3/b", "generator/encPhy/m1fc2/fc", "generator/buildPhy/m1fc16/fc", "generator/buildPhy/d16/w", "generator/buildPhy/d16/b", "generator/buildPhy/energy-d2/w", "generator/buildPhy/energy-d2/b", "generator/buildPhy/energy-d1/w", "generator/buildPhy/energy-d1/b", "generator/R128_4/res1/w", "generator/R128_4/res1/instance_norm/shift", "generator/R128_4/res1/instance_norm/scale", "generator/R128_4/res2/w", "generator/R128_4/res2/instance_norm/shift", "generator/R128_4/res2/instance_norm/scale", "generator/R128_5/res1/w", "generator/R128_5/res1/instance_norm/shift", "generator/R128_5/res1/instance_norm/scale", "generator/R128_5/res2/w", "generator/R128_5/res2/instance_norm/shift", "generator/R128_5/res2/instance_norm/scale", "generator/vort_u32/w", "generator/vort_u32/b", "generator/vort_u32/instance_norm/shift", "generator/vort_u32/instance_norm/scale", "generator/vort_u16/w", "generator/vort_u16/b", "generator/vort_u16/instance_norm/shift", "generator/vort_u16/instance_norm/scale", "generator/vort_v1/w", "generator/vort_v1/b", "generator/vein_16/w", "generator/vein_16/b", "generator/vein_16/instance_norm/shift", "generator/vein_16/instance_norm/scale", "generator/vein_24/w", "generator/vein_24/b", "generator/vein_24/instance_norm/shift", "generator/vein_24/instance_norm/scale", "generator/vein_32/w", "generator/vein_32/b", "generator/vein_32/instance_norm/shift", "generator/vein_32/instance_norm/scale", "generator/u64/w", "generator/u64/b", "generator/u64/instance_norm/shift", "generator/u64/instance_norm/scale", "generator/u32/w", "generator/u32/b", "generator/u32/instance_norm/shift", "generator/u32/instance_norm/scale", "generator/c7s1-3/w", "generator/c7s1-3/b", "generator/zoom_out/HS-8/w", "generator/zoom_out/HS-8/b", "generator/zoom_out/HS-out/w", "generator/zoom_out/HS-out/b"
        ]
        if not withZoom: namelist = namelist[4:-4]

        if not os.path.exists(ckpt):
            print(ckpt, "does not exist. Try to download d2v model for training...")
            download_file(ckpt, "https://rachelcmy.github.io/pinf_smoke/data/d2v_3Dmodel.npz")
        
        self.var_dict = np.load(ckpt, allow_pickle=True)["arr_0"].item()
        var_list = [self.var_dict[k] for k in namelist]
        self.fix_layer_tool = FixLayer(var_list, namelist)
        
        self.dim = 2 if FLAGS.is2D else 3
        tarsize = FLAGS.crop_size * int( max(FLAGS.zoom_factor,1) )
        self.input_shape = [1] + [1] + [256 if withZoom else 64]*3
        self.output_shape = [tarsize] * (2 if FLAGS.is2D else 3)
        self._norm='instance'
        self._activation='relu'
        
        # start to build the model.
        C1 = MyConvBlock(self.fix_layer_tool, 1, 32, 'c7s1-32', 7, 1, 
            pad='reflect',is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
        C2 = MyConvBlock(self.fix_layer_tool, 32, 64, 'd64', 3, 2, 
            pad='same',is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
        C3 = MyConvBlock(self.fix_layer_tool, 64, 128, 'd128', 3, 2, 
            pad='same',is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
        
        self.Cin_list = [C1,C2,C3]
        res_ch = 128
        self.Res_list = [
             MyResBlock(self.fix_layer_tool, res_ch, 'R128_{}'.format(i), 
                pad='reflect',is2D=FLAGS.is2D,norm=self._norm)
            for i in range(FLAGS.num_resblock//2+1)
        ]
        self.encode_param_layer(FLAGS, res_ch)
        self.build_param_layer(FLAGS)
        if FLAGS.useEnergy: 
            res_ch = res_ch + 1
        res_ch = res_ch + FLAGS.mid_ch
        self.Res_list = self.Res_list + [
             MyResBlock(self.fix_layer_tool, res_ch, 'R128_{}'.format(i), 
                pad='reflect',is2D=FLAGS.is2D,norm=self._norm)
            for i in range(FLAGS.num_resblock//2+1, FLAGS.num_resblock)
        ]
        if FLAGS.useVortEnd:
            vort_lvl1 = MyDeconvBlock(self.fix_layer_tool, res_ch, 32, 'vort_u32', 3, 2, 
                is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
            vort_lvl2 = MyDeconvBlock(self.fix_layer_tool, 32, 16, 'vort_u16', 3, 2, 
                is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
            enc_vortEnd = MyConvBlock(self.fix_layer_tool, 16, 1 if FLAGS.is2D else 3, 'vort_v1', 7, 1, 
                pad='reflect',is2D=FLAGS.is2D, norm=None, activation=None)
            self.VORT_list = [vort_lvl1,vort_lvl2,enc_vortEnd]
            # ve_in = make_ve_in()
            vein_lvl2 = MyConvBlock(self.fix_layer_tool, 2 if FLAGS.is2D else 6, 16,  'vein_16', 7, 1, 
                pad='reflect',is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
            vein_lvl1 = MyConvBlock(self.fix_layer_tool, 16, 24,  'vein_24', 3, 2, 
                pad='same',is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
            vein_lvl = MyConvBlock(self.fix_layer_tool, 24, 32,  'vein_32', 3, 2, 
                pad='same',is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
            self.VORTin_list = [vein_lvl2,vein_lvl1,vein_lvl]
            res_ch = res_ch + 32
            # G = tf.concat([G, vein_lvl], axis = -1)
        Out1 = MyDeconvBlock(self.fix_layer_tool, res_ch, 64, 'u64', 3, 2, 
                is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
        Out2 = MyDeconvBlock(self.fix_layer_tool, 64, 32, 'u32', 3, 2, 
                is2D=FLAGS.is2D, norm=self._norm, activation=self._activation)
        out_ch = 1 if FLAGS.is2D else 3
        Out3 = MyConvBlock(self.fix_layer_tool, 32, out_ch, 'c7s1-3', 7, 1, 
                pad='reflect',is2D=FLAGS.is2D, norm=None, activation=None)
        self.Out_list = [Out1,Out2,Out3]
        
            
        self.FLAGS = FLAGS
        

    def forward(self, den_input):
        dim = 2 if self.FLAGS.is2D else 3
        pool_k = [self.FLAGS.crop_size//4//8] * dim
        pool_fn = nn.AvgPool2d if self.FLAGS.is2D else nn.AvgPool3d
        param_n = self.FLAGS.phy_len
        mid_ch = self.FLAGS.mid_ch
        mid_shape = [mid_ch]+[8]*dim
        permute_mid_shape = [8]*dim + [mid_ch]
        mid_repeat = self.FLAGS.crop_size//4//8

        phy_input = torch.Tensor([[self.FLAGS.buoy, float(self.FLAGS.OpenBounds)]])
        
        g = den_input
        for Cin in self.Cin_list:
            g = Cin(g)
            
        for i in range(self.FLAGS.num_resblock):
            g = self.Res_list[i](g)
            if (i == self.FLAGS.num_resblock // 2):
                phy_g = self.PHY_list[0](g)
                if self.FLAGS.useEnergy: 
                    KE_g = phy_g
                    for KE_fn in self.KE_list:
                        KE_g = KE_fn(KE_g)
                # mid_in FLAGS.batch_size x m, 8x8x16
                phy_g = pool_fn(pool_k, stride=pool_k, padding=0)(phy_g)
                phy_g = self.PHY_list[1](phy_g)
                phy_g = phy_g.permute([0,2,3,4,1])
                phy_g = torch.reshape(phy_g, (self.FLAGS.batch_size, -1) )
                phy_g = self.PHY_list[2](phy_g)

                phy_in = phy_input.expand(phy_g.shape)
                phy_in = torch.cat([phy_g, phy_in], dim=1)
                
                if self.FLAGS.useEnergy: 
                    KE_in = torch.ones_like(KE_g) * -1.0
                    KE_in = torch.cat([KE_g, KE_in], dim=1)

                phy_g = self.PHYin_list[0](phy_in)
                phy_g = phy_g.view([-1]+permute_mid_shape) # bx16x8x8(x8)
                phy_g = phy_g.permute([0,4,1,2,3])
                phy_g = self.PHYin_list[1](phy_g)
                phy_g = torch.repeat_interleave(phy_g, mid_repeat, dim=2)
                phy_g = torch.repeat_interleave(phy_g, mid_repeat, dim=3)
                if dim == 3:
                    phy_g = torch.repeat_interleave(phy_g, mid_repeat, dim=4)
                if self.FLAGS.useEnergy: 
                    KE_g = self.KEin_list[0](KE_in)
                    KE_g = self.KEin_list[1](KE_g)
                    KE_g = torch.repeat_interleave(KE_g, 4, dim=2)
                    KE_g = torch.repeat_interleave(KE_g, 4, dim=3)
                    if dim == 3:
                        KE_g = torch.repeat_interleave(KE_g, 4, dim=4)
                    phy_g = torch.cat([phy_g, KE_g], dim=1)
                g = torch.cat([g, phy_g], dim=1)
        # vort
        if self.FLAGS.useVortEnd:
            v_g = g
            for v_fn in self.VORT_list:
                v_g = v_fn(v_g)
            ve_in = torch.ones_like(v_g) * -10.0
            v_g = torch.cat([v_g, ve_in], dim=1)
            
            for v_fn in self.VORTin_list:
                v_g = v_fn(v_g)
            g = torch.cat([g, v_g], dim=1)
        for out_fn in self.Out_list:
            g = out_fn(g)
        # from NCHW to NHWC
        permute_order = [0,2,3,1]
        if not self.FLAGS.is2D:
            permute_order = [0,2,3,4,1]
        g = g.permute(permute_order)
        # curl in NHWC mode, return NHW3 as velocity
        g = (curl2D(g) if dim == 2 else curl3D(g))
        return g