

import numpy as np
import sys, os
import imageio
# torch.autograd.set_detect_anomaly(True)
import torch, torchvision
from torch import optim, nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF
import cv2 as cv
#####################################################################
# custom Logger to write Log to file
class Logger(object):
    def __init__(self, summary_dir, silent=False, fname="logfile.txt"):
        self.terminal = sys.stdout
        self.silent = silent
        self.log = open(os.path.join(summary_dir, fname), "a") 
        cmdline = " ".join(sys.argv)+"\n"
        self.log.write(cmdline) 
    def write(self, message):
        if not self.silent: 
            self.terminal.write(message)
        self.log.write(message) 
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def printENV():
    check_list = ['CUDA_VISIBLE_DEVICES']
    for name in check_list:
        if name in os.environ:
            print(name, os.environ[name])
        else:
            print(name, "Not find")

    sys.stdout.flush()


#####################################################################
# Visualization Tools

def velLegendHSV(hsvin, is3D, lw=-1, constV=255):
    # hsvin: (b), h, w, 3
    # always overwrite hsvin borders [lw], please pad hsvin before hand
    # or fill whole hsvin (lw < 0)
    ih, iw = hsvin.shape[-3:-1]
    if lw<=0: # fill whole
        a_list, b_list = [range(ih)], [range(iw)]
    else: # fill border
        a_list = [range(ih),  range(lw), range(ih), range(ih-lw, ih)]
        b_list = [range(lw),  range(iw), range(iw-lw, iw), range(iw)]
    for a,b in zip(a_list, b_list):
        for _fty in a:
            for _ftx in b:
                fty = _fty - ih//2
                ftx = _ftx - iw//2
                ftang = np.arctan2(fty, ftx) + np.pi
                ftang = ftang*(180/np.pi/2)
                # print("ftang,min,max,mean", ftang.min(), ftang.max(), ftang.mean())
                # ftang,min,max,mean 0.7031249999999849 180.0 90.3515625
                hsvin[...,_fty,_ftx,0] = np.expand_dims(ftang, axis=-1) # 0-360 
                # hsvin[...,_fty,_ftx,0] = ftang
                hsvin[...,_fty,_ftx,2] = constV
                if (not is3D) or (lw == 1):
                    hsvin[...,_fty,_ftx,1] = 255
                else:
                    thetaY1 = 1.0 - ((ih//2) - abs(fty)) / float( lw if (lw > 1) else (ih//2) )
                    thetaY2 = 1.0 - ((iw//2) - abs(ftx)) / float( lw if (lw > 1) else (iw//2) )
                    fthetaY = max(thetaY1, thetaY2) * (0.5*np.pi)
                    ftxY, ftyY = np.cos(fthetaY), np.sin(fthetaY)
                    fangY = np.arctan2(ftyY, ftxY)
                    fangY = fangY*(240/np.pi*2) # 240 - 0
                    hsvin[...,_fty,_ftx,1] = 255 - fangY
                    # print("fangY,min,max,mean", fangY.min(), fangY.max(), fangY.mean())
    # finished velLegendHSV.

def cubecenter(cube, axis, half = 0):
    # cube: (b,)h,h,h,c
    # axis: 1 (z), 2 (y), 3 (x)
    reduce_axis = [a for a in [1,2,3] if a != axis]
    pack = np.mean(cube, axis=tuple(reduce_axis)) # (b,)h,c
    pack = np.sqrt(np.sum( np.square(pack), axis=-1 ) + 1e-6) # (b,)h

    length = cube.shape[axis-5] # h
    weights = np.arange(0.5/length,1.0,1.0/length)
    if half == 1: # first half
        weights = np.where( weights < 0.5, weights, np.zeros_like(weights))
        pack = np.where( weights < 0.5, pack, np.zeros_like(pack))
    elif half == 2: # second half
        weights = np.where( weights > 0.5, weights, np.zeros_like(weights))
        pack = np.where( weights > 0.5, pack, np.zeros_like(pack))

    weighted = pack * weights # (b,)h
    weiAxis = np.sum(weighted, axis=-1) / np.sum(pack, axis=-1) * length # (b,)
    
    return weiAxis.astype(np.int32) # a ceiling is included

def vel2hsv(velin, is3D, logv, scale=None): # 2D
    fx, fy = velin[...,0], velin[...,1]
    ori_shape = list(velin.shape[:-1]) + [3]
    if is3D: 
        fz = velin[...,2]
        ang = np.arctan2(fz, fx) + np.pi # angXZ
        zxlen2 = fx*fx+fz*fz
        angY = np.arctan2(np.abs(fy), np.sqrt(zxlen2))
        v = np.sqrt(zxlen2+fy*fy)
    else:
        v = np.sqrt(fx*fx+fy*fy)
        ang = np.arctan2(fy, fx) + np.pi
    
    if logv:
        v = np.log10(v+1)
    
    hsv = np.zeros(ori_shape, np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    if is3D:
        hsv[...,1] = 255 - angY*(240/np.pi*2)  
    else:
        hsv[...,1] = 255
    if scale is not None:
        hsv[...,2] = np.minimum(v*scale, 255)
    else:
        hsv[...,2] = v/max(v.max(),1e-6) * 255.0
    return hsv


def vel_uv2hsv(vel, scale = 160, is3D=False, logv=False, mix=False):
    # vel: a np.float32 array, in shape of (?=b,) d,h,w,3 for 3D and (?=b,)h,w, 2 or 3 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good. 
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use more slices to get a volumetric visualization if True, which is slow

    ori_shape = list(vel.shape[:-1]) + [3] # (?=b,) d,h,w,3
    if is3D: 
        new_range = list( range( len(ori_shape) ) )
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXvel = np.transpose(vel, z_new_range)
        
        _xm,_ym,_zm = (ori_shape[-2]-1)//2, (ori_shape[-3]-1)//2, (ori_shape[-4]-1)//2
        
        if mix:
            _xlist = [cubecenter(vel, 3, 1),_xm,cubecenter(vel, 3, 2)]
            _ylist = [cubecenter(vel, 2, 1),_ym,cubecenter(vel, 2, 2)]
            _zlist = [cubecenter(vel, 1, 1),_zm,cubecenter(vel, 1, 2)]
        else:
            _xlist, _ylist, _zlist = [_xm], [_ym], [_zm]

        hsv = []
        for _x, _y, _z in zip (_xlist, _ylist, _zlist):
            # print(_x, _y, _z)
            _x, _y, _z = np.clip([_x, _y, _z], 0, ori_shape[-2:-5:-1])
            _yz = YZXvel[...,_x,:]
            _yz = np.stack( [_yz[...,2],_yz[...,0],_yz[...,1]], axis=-1)
            _yx = YZXvel[...,_z,:,:]
            _yx = np.stack( [_yx[...,0],_yx[...,2],_yx[...,1]], axis=-1)
            _zx = YZXvel[...,_y,:,:,:]
            _zx = np.stack( [_zx[...,0],_zx[...,1],_zx[...,2]], axis=-1)
            # print(_yx.shape, _yz.shape, _zx.shape)

            # in case resolution is not a cube, (res,res,res)
            _yxz = np.concatenate( [ #yz, yx, zx
                _yx, _yz ], axis = -2) # (?=b,),h,w+zdim,3
            
            if ori_shape[-3] < ori_shape[-4]:
                pad_shape = list(_yxz.shape) #(?=b,),h,w+zdim,3
                pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
                _pad = np.zeros(pad_shape, dtype=np.float)
                _yxz = np.concatenate( [_yxz,_pad], axis = -3)
            elif ori_shape[-3] > ori_shape[-4]:
                pad_shape = list(_zx.shape) #(?=b,),h,w+zdim,3
                pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

                _zx = np.concatenate( 
                    [_zx,np.zeros(pad_shape, dtype=np.float)], axis = -3)
            
            midVel = np.concatenate( [ #yz, yx, zx
                _yxz, _zx
            ], axis = -2) # (?=b,),h,w*3,3
            hsv += [vel2hsv(midVel, True, logv, scale)]
        # remove depth dim, increase with zyx slices
        ori_shape[-3] = 3 * ori_shape[-2]
        ori_shape[-2] = ori_shape[-1]
        ori_shape = ori_shape[:-1]
    else:
        hsv = [vel2hsv(vel, False, logv, scale)]

    bgr = []
    for _hsv in hsv:
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape([-1]+ori_shape[-2:])
        if is3D:
            velLegendHSV(_hsv, is3D, lw=max(1,min(6,int(0.025*ori_shape[-2]))), constV=255)
        _hsv = cv.cvtColor(_hsv, cv.COLOR_HSV2BGR)
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape(ori_shape)
        bgr += [_hsv]
    if len(bgr) == 1:
        bgr = bgr[0]
    else:
        bgr = bgr[0] * 0.2 + bgr[1] * 0.6 + bgr[2] * 0.2
    return bgr.astype(np.uint8)[::-1] # flip Y


def den_scalar2rgb(den, scale=160, is3D=False, logv=False, mix=True):
    # den: a np.float32 array, in shape of (?=b,) d,h,w,1 for 3D and (?=b,)h,w,1 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good. 
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use averaged value as a volumetric visualization if True, else show middle slice

    ori_shape = list(den.shape)
    if ori_shape[-1] != 1:
        ori_shape.append(1)
        den = np.reshape(den, ori_shape)

    if is3D: 
        new_range = list( range( len(ori_shape) ) )
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXden = np.transpose(den, z_new_range)
                
        if not mix:
            _yz = YZXden[...,(ori_shape[-2]-1)//2,:]
            _yx = YZXden[...,(ori_shape[-4]-1)//2,:,:]
            _zx = YZXden[...,(ori_shape[-3]-1)//2,:,:,:]
        else:
            _yz = np.average(YZXden, axis=-2)
            _yx = np.average(YZXden, axis=-3)
            _zx = np.average(YZXden, axis=-4)
            # print(_yx.shape, _yz.shape, _zx.shape)

        # in case resolution is not a cube, (res,res,res)
        _yxz = np.concatenate( [ #yz, yx, zx
            _yx, _yz ], axis = -2) # (?=b,),h,w+zdim,1
        
        if ori_shape[-3] < ori_shape[-4]:
            pad_shape = list(_yxz.shape) #(?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
            _pad = np.zeros(pad_shape, dtype=np.float)
            _yxz = np.concatenate( [_yxz,_pad], axis = -3)
        elif ori_shape[-3] > ori_shape[-4]:
            pad_shape = list(_zx.shape) #(?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

            _zx = np.concatenate( 
                [_zx,np.zeros(pad_shape, dtype=np.float)], axis = -3)
        
        midDen = np.concatenate( [ #yz, yx, zx
            _yxz, _zx
        ], axis = -2) # (?=b,),h,w*3,1
    else:
        midDen = den

    if logv:
        midDen = np.log10(midDen+1)
    if scale is None:
        midDen = midDen / max(midDen.max(),1e-6) * 255.0
    else:
        midDen = midDen * scale
    grey = np.clip(midDen, 0, 255)

    return grey.astype(np.uint8)[::-1] # flip y


#####################################################################
# Physics Tools

def jacobian3D(x):
    # x, (b,)d,h,w,ch, pytorch tensor
    # return jacobian and curl

    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = torch.cat((dudx, torch.unsqueeze(dudx[:,:,:,-1], 3)), 3)
    dvdx = torch.cat((dvdx, torch.unsqueeze(dvdx[:,:,:,-1], 3)), 3)
    dwdx = torch.cat((dwdx, torch.unsqueeze(dwdx[:,:,:,-1], 3)), 3)

    dudy = torch.cat((dudy, torch.unsqueeze(dudy[:,:,-1,:], 2)), 2)
    dvdy = torch.cat((dvdy, torch.unsqueeze(dvdy[:,:,-1,:], 2)), 2)
    dwdy = torch.cat((dwdy, torch.unsqueeze(dwdy[:,:,-1,:], 2)), 2)

    dudz = torch.cat((dudz, torch.unsqueeze(dudz[:,-1,:,:], 1)), 1)
    dvdz = torch.cat((dvdz, torch.unsqueeze(dvdz[:,-1,:,:], 1)), 1)
    dwdz = torch.cat((dwdz, torch.unsqueeze(dwdz[:,-1,:,:], 1)), 1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = torch.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], -1)
    c = torch.stack([u,v,w], -1)
    
    return j, c

def curl2D(x, data_format='NHWC'):
    assert data_format == 'NHWC'
    u = x[:,1:,:,0] - x[:,:-1,:,0] # ds/dy
    v = x[:,:,:-1,0] - x[:,:,1:,0] # -ds/dx,
    u = torch.cat([u, u[:,-1:,:]], dim=1)
    v = torch.cat([v, v[:,:,-1:]], dim=2)
    c = tf.stack([u,v], dim=-1)
    return c

def curl3D(x, data_format='NHWC'):
    assert data_format == 'NHWC'
    # x: bzyxc
    # dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx   = x[:,:,:,1:,1] - x[:,:,:,:-1,1] #
    dwdx   = x[:,:,:,1:,2] - x[:,:,:,:-1,2] #
    dudy   = x[:,:,1:,:,0] - x[:,:,:-1,:,0] # 
    # dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy   = x[:,:,1:,:,2] - x[:,:,:-1,:,2] #
    dudz   = x[:,1:,:,:,0] - x[:,:-1,:,:,0] # 
    dvdz   = x[:,1:,:,:,1] - x[:,:-1,:,:,1] # 
    # dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # dudx = torch.cat((dudx, dudx[:,:,:,-1]), dim=3)
    dvdx   = torch.cat((dvdx, dvdx[:,:,:,-1:]), dim=3) #
    dwdx   = torch.cat((dwdx, dwdx[:,:,:,-1:]), dim=3) #

    dudy   = torch.cat((dudy, dudy[:,:,-1:,:]), dim=2) #
    # dvdy = torch.cat((dvdy, dvdy[:,:,-1:,:]), dim=2)
    dwdy   = torch.cat((dwdy, dwdy[:,:,-1:,:]), dim=2) # 

    dudz   = torch.cat((dudz, dudz[:,-1:,:,:]), dim=1) #
    dvdz   = torch.cat((dvdz, dvdz[:,-1:,:,:]), dim=1) # 
    # dwdz = torch.cat((dwdz, dwdz[:,-1:,:,:]), dim=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    # j = tf.stack([
    #       dudx,dudy,dudz,
    #       dvdx,dvdy,dvdz,
    #       dwdx,dwdy,dwdz
    # ], dim=-1)
    # curl = dwdy-dvdz,dudz-dwdx,dvdx-dudy
    c = torch.stack([u,v,w], dim=-1)
    
    return c

def jacobian3D_np(x):
    # x, (b,)d,h,w,ch
    # return jacobian and curl

    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = np.concatenate((dudy, np.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = np.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
    c = np.stack([u,v,w], axis=-1)
    
    return j, c


# from FFJORD github code
def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac

# from FFJORD github code
def divergence_exact(input_points, outputs):
    # requires three backward passes instead one like divergence_approx
    jac = _get_minibatch_jacobian(outputs, input_points)
    diagonal = jac.view(jac.shape[0], -1)[:, :: (jac.shape[1]+1)]
    return torch.sum(diagonal, 1)


def PDE_EQs(D_t, D_x, D_y, D_z, U, U_t=None, U_x=None, U_y=None, U_z=None):
    eqs = []
    dts = [D_t] 
    dxs = [D_x] 
    dys = [D_y] 
    dzs = [D_z] 

    if None not in [U_t, U_x, U_y, U_z]:
        dts += U_t.split(1, dim = -1) # [d_t, u_t, v_t, w_t] # (N,1)
        dxs += U_x.split(1, dim = -1) # [d_x, u_x, v_x, w_x]
        dys += U_y.split(1, dim = -1) # [d_y, u_y, v_y, w_y]
        dzs += U_z.split(1, dim = -1) # [d_z, u_z, v_z, w_z]
        
    u,v,w = U.split(1, dim=-1) # (N,1)
    for dt, dx, dy, dz in zip (dts, dxs, dys, dzs):
        _e = dt + (u*dx + v*dy + w*dz)
        eqs += [_e]
    # transport and nse equations:
    # e1 = d_t + (u*d_x + v*d_y + w*d_z) - PecInv*(c_xx + c_yy + c_zz)          , should = 0
    # e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - ReyInv*(u_xx + u_yy + u_zz)    , should = 0
    # e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - ReyInv*(v_xx + v_yy + v_zz)    , should = 0
    # e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - ReyInv*(w_xx + w_yy + w_zz)    , should = 0
    # e5 = u_x + v_y + w_z                                                      , should = 0
    # For simplification, we assume PecInv = 0.0, ReyInv = 0.0, pressure p = (0,0,0)                      
    
    if None not in [U_t, U_x, U_y, U_z]:
        # eqs += [ u_x + v_y + w_z ]
        eqs += [ dxs[1] + dys[2] + dzs[3] ]

    if True: # scale regularization
        eqs += [ (u*u + v*v + w*w)* 1e-1]
    
    return eqs

#####################################################################
# Coord Tools (all for torch Tensors)
# Coords:
# 1. resolution space, Frames x Depth x H x W, coord (frame_t, voxel_z, voxel_y, voxel_x),
# 2. simulation space, scale the resolution space to around 0-1, 
#    (FrameLength and Width in [0-1], Height and Depth keep ratios wrt Width)
# 3. target space, 
# 4. world space,
# 5. camera spaces,

# Vworld, Pworld; velocity, position in 4. world coord.
# Vsmoke, Psmoke; velocity, position in 2. simulation coord.
# w2s, 4.world to 3.target matrix (vel transfer uses rotation only; pos transfer includes offsets)
# s2w, 3.target to 4.world matrix (vel transfer uses rotation only; pos transfer includes offsets)
# scale_vector, to scale from 2.simulation space to 3.target space (no rotation, no offset)
#        for synthetic data, scale_vector = openvdb voxel size * [W,H,D] grid resolution (x first, z last), 
#        for e.g., scale_vector = 0.0469 * 256 = 12.0064
# st_factor, spatial temporal resolution ratio, to scale velocity from 2.simulation unit to 1.resolution unit
#        for e.g.,  st_factor = [W/float(max_timestep),H/float(max_timestep),D/float(max_timestep)]

# functions to transfer between 4. world space and 2. simulation space, 
# velocity are further scaled according to resolution as in mantaflow
def vel_world2smoke(Vworld, w2s, scale_vector, st_factor):
    _st_factor = torch.Tensor(st_factor).expand((3, ))
    vel_rot = Vworld[..., None, :] * (w2s[:3,:3])
    vel_rot = torch.sum(vel_rot, -1) # 4.world to 3.target 
    vel_scale = vel_rot / (scale_vector) * _st_factor # 3.target to 2.simulation
    return vel_scale

def vel_smoke2world(Vsmoke, s2w, scale_vector, st_factor):
    _st_factor = torch.Tensor(st_factor).expand((3, ))
    vel_scale = Vsmoke * (scale_vector) / _st_factor # 2.simulation to 3.target
    vel_rot = torch.sum(vel_scale[..., None, :] * (s2w[:3,:3]), -1) # 3.target to 4.world
    return vel_rot
    
def pos_world2smoke(Pworld, w2s, scale_vector):
    pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3,:3]), -1) # 4.world to 3.target 
    pos_off = (w2s[:3, -1]).expand(pos_rot.shape) # 4.world to 3.target 
    new_pose = pos_rot + pos_off 
    pos_scale = new_pose / (scale_vector) # 3.target to 2.simulation
    return pos_scale

def off_smoke2world(Offsmoke, s2w, scale_vector):
    off_scale = Offsmoke * (scale_vector) # 2.simulation to 3.target
    off_rot = torch.sum(off_scale[..., None, :] * (s2w[:3,:3]), -1)  # 3.target to 4.world
    return off_rot

def pos_smoke2world(Psmoke, s2w, scale_vector):
    pos_scale = Psmoke * (scale_vector) # 2.simulation to 3.target
    pos_rot = torch.sum(pos_scale[..., None, :] * (s2w[:3,:3]), -1)  # 3.target to 4.world
    pos_off = (s2w[:3, -1]).expand(pos_rot.shape) # 3.target to 4.world
    return pos_rot+pos_off

def get_voxel_pts(H, W, D, s2w, scale_vector, n_jitter=0, r_jitter=0.8):
    """Get voxel positions."""
    
    i, j, k = torch.meshgrid(torch.linspace(0, D-1, D),
                       torch.linspace(0, H-1, H),
                       torch.linspace(0, W-1, W))
    pts = torch.stack([(k+0.5)/W, (j+0.5)/H, (i+0.5)/D], -1) 
    # shape D*H*W*3, value [(x,y,z)] , range [0,1]

    jitter_r = torch.Tensor([r_jitter/W,r_jitter/H,r_jitter/D]).float().expand(pts.shape)
    for i_jitter in range(n_jitter):
        off_i = torch.rand(pts.shape, dtype=torch.float)-0.5
        # shape D*H*W*3, value [(x,y,z)] , range [-0.5,0.5]

        pts = pts + off_i * jitter_r

    return pos_smoke2world(pts, s2w, scale_vector)


def get_voxel_pts_offset(H, W, D, s2w, scale_vector, r_offset=0.8):
    """Get voxel positions."""
    
    i, j, k = torch.meshgrid(torch.linspace(0, D-1, D),
                       torch.linspace(0, H-1, H),
                       torch.linspace(0, W-1, W))
    pts = torch.stack([(k+0.5)/W, (j+0.5)/H, (i+0.5)/D], -1) 
    # shape D*H*W*3, value [(x,y,z)] , range [0,1]

    jitter_r = torch.Tensor([r_offset/W,r_offset/H,r_offset/D]).expand(pts.shape)
    off_i = torch.rand([1,1,1,3], dtype=torch.float)-0.5
    # shape 1*1*1*3, value [(x,y,z)] , range [-0.5,0.5]
    pts = pts + off_i * jitter_r

    return pos_smoke2world(pts, s2w, scale_vector)

class BBox_Tool(object):
    def __init__(self, smoke_tran_inv, smoke_scale, in_min=0.0, in_max=1.0):
        self.s_w2s = torch.Tensor(smoke_tran_inv).expand([4,4])
        self.s_scale = torch.Tensor(smoke_scale).expand([3])
        self.s_min = torch.Tensor(in_min).expand([3])
        self.s_max = torch.Tensor(in_max).expand([3])

    def setMinMax(self, in_min=0.0, in_max=1.0):
        self.s_min = torch.Tensor(in_min).expand([3])
        self.s_max = torch.Tensor(in_max).expand([3])

    def isInside(self, inputs_pts):
        target_pts = pos_world2smoke(inputs_pts, self.s_w2s, self.s_scale)
        above = torch.logical_and(target_pts[...,0] >= self.s_min[0], target_pts[...,1] >= self.s_min[1] ) 
        above = torch.logical_and(above, target_pts[...,2] >= self.s_min[2] ) 
        below = torch.logical_and(target_pts[...,0] <= self.s_max[0], target_pts[...,1] <= self.s_max[1] ) 
        below = torch.logical_and(below, target_pts[...,2] <= self.s_max[2] ) 
        outputs = torch.logical_and(below, above)
        return outputs

    def insideMask(self, inputs_pts):
        return self.isInside(inputs_pts).to(torch.float)
    

class Voxel_Tool(object):

    def __get_tri_slice(self, _xm, _ym, _zm, _n=1):
        _yz = torch.reshape(self.pts[...,_xm:_xm+_n,:],(-1,3))
        _zx = torch.reshape(self.pts[:,_ym:_ym+_n,...],(-1,3))
        _xy = torch.reshape(self.pts[_zm:_zm+_n,...],(-1,3))
        
        pts_mid = torch.cat([_yz, _zx, _xy], dim=0)
        npMaskXYZ = [np.zeros([self.D,self.H,self.W,1], dtype=np.float32) for _ in range(3)]
        npMaskXYZ[0][...,_xm:_xm+_n,:] = 1.0
        npMaskXYZ[1][:,_ym:_ym+_n,...] = 1.0
        npMaskXYZ[2][_zm:_zm+_n,...] = 1.0
        return pts_mid, torch.tensor(np.clip(npMaskXYZ[0]+npMaskXYZ[1]+npMaskXYZ[2], 1e-6, 3.0))

    def __pad_slice_to_volume(self, _slice, _n, mode=0):
        # mode: 0, x_slice, 1, y_slice, 2, z_slice
        tar_shape = [self.D,self.H,self.W]
        in_shape = tar_shape[:]
        in_shape[-1-mode] = _n
        fron_shape = tar_shape[:]
        fron_shape[-1-mode] = (tar_shape[-1-mode] - _n)//2
        back_shape = tar_shape[:]
        back_shape[-1-mode] = (tar_shape[-1-mode] - _n - fron_shape[-1-mode])

        cur_slice = _slice.view(in_shape+[-1])
        front_0 = torch.zeros(fron_shape + [cur_slice.shape[-1]])
        back_0 = torch.zeros(back_shape + [cur_slice.shape[-1]])

        volume = torch.cat([front_0, cur_slice, back_0], dim=-2-mode)
        return volume


    def __init__(self, smoke_tran, smoke_tran_inv, smoke_scale, D, H, W, middleView=None):
        self.s_s2w = torch.Tensor(smoke_tran).expand([4,4])
        self.s_w2s = torch.Tensor(smoke_tran_inv).expand([4,4])
        self.s_scale = torch.Tensor(smoke_scale).expand([3])
        self.D = D
        self.H = H
        self.W = W
        self.pts = get_voxel_pts(H, W, D, self.s_s2w, self.s_scale)
        self.pts_mid = None
        self.npMaskXYZ = None
        self.middleView = middleView
        if middleView is not None:
            _n = 1 if self.middleView=="mid" else 3
            _xm,_ym,_zm = (W-_n)//2, (H-_n)//2, (D-_n)//2
            self.pts_mid, self.npMaskXYZ = self.__get_tri_slice(_xm,_ym,_zm,_n)

    def get_raw_at_pts(self, cur_pts, chunk=1024*32, use_viewdirs=False, 
        network_query_fn=None, network_fn=None):
        input_shape = list(cur_pts.shape[0:-1])

        pts_flat = cur_pts.view(-1, 4)
        pts_N = pts_flat.shape[0]
        # Evaluate model
        all_raw = []
        viewdir_zeros = torch.zeros([chunk,3],dtype=torch.float) if use_viewdirs else None
        for i in range(0, pts_N, chunk):
            pts_i = pts_flat[i:i+chunk]
            viewdir_i = viewdir_zeros[:pts_i.shape[0]] if use_viewdirs else None

            raw_i = network_query_fn(pts_i, viewdir_i, network_fn) 
            all_raw.append(raw_i)

        raw = torch.cat(all_raw, 0).view(input_shape+[-1])
        return raw

    def get_density_flat(self, cur_pts, chunk=1024*32, use_viewdirs=False, 
        network_query_fn=None, network_fn=None, getStatic=True):
        flat_raw = self.get_raw_at_pts(cur_pts, chunk, use_viewdirs, network_query_fn, network_fn)
        den_raw = F.relu(flat_raw[...,-1:])
        returnStatic = getStatic and (flat_raw.shape[-1] > 4)
        if returnStatic:
            static_raw = F.relu(flat_raw[...,3:4])
            return [den_raw, static_raw]
        return [den_raw]

    def get_velocity_flat(self, cur_pts, batchify_fn,chunk=1024*32, 
        vel_model=None):
        pts_N = cur_pts.shape[0]
        world_v = []
        for i in range(0, pts_N, chunk):
            input_i = cur_pts[i:i+chunk]
            vel_i = batchify_fn(vel_model, chunk)(input_i)
            world_v.append(vel_i)
        world_v = torch.cat(world_v, 0)
        return world_v

    def get_density_and_derivatives(self, cur_pts, chunk=1024*32, use_viewdirs=False, 
        network_query_fn=None, network_fn=None):
        _den = self.get_density_flat(cur_pts, chunk, use_viewdirs,network_query_fn, network_fn, False)[0]
        # requires 1 backward passes 
        # The minibatch Jacobian matrix of shape (N, D_y=1, D_x=4)
        jac = _get_minibatch_jacobian(_den, cur_pts)
        _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,1)
        return _den, _d_x, _d_y, _d_z, _d_t

    def get_velocity_and_derivatives(self, cur_pts, chunk=1024*32, batchify_fn=None, vel_model=None):
        _vel = self.get_velocity_flat(cur_pts, batchify_fn, chunk, vel_model)
        # requires 3 backward passes
        # The minibatch Jacobian matrix of shape (N, D_y=3, D_x=4)
        jac = _get_minibatch_jacobian(_vel, cur_pts)
        _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,3)
        return _vel, _u_x, _u_y, _u_z, _u_t
        
    def get_voxel_density_list(self,t=None,chunk=1024*32, use_viewdirs=False, 
        network_query_fn=None, network_fn=None, middle_slice=False):
        D,H,W = self.D,self.H,self.W
        # middle_slice, only for fast visualization of the middle slice
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]
        if t is not None:
            input_t = torch.ones([pts_N, 1])*float(t)
            pts_flat = torch.cat([pts_flat,input_t], dim=-1)

        den_list = self.get_density_flat(pts_flat, chunk, use_viewdirs, network_query_fn, network_fn)

        return_list = []
        for den_raw in den_list:
            if middle_slice:
                # only for fast visualization of the middle slice
                _n = 1 if self.middleView=="mid" else 3
                _yzV, _zxV, _xyV = torch.split(den_raw, [D*H*_n,D*W*_n,H*W*_n], dim=0)
                mixV = self.__pad_slice_to_volume(_yzV, _n, 0) + self.__pad_slice_to_volume(_zxV, _n, 1) +self.__pad_slice_to_volume(_xyV, _n, 2)
                return_list.append(mixV / self.npMaskXYZ)
            else:
                return_list.append(den_raw.view(D,H,W,1))
        return return_list
        
    def get_voxel_velocity(self,deltaT,t,batchify_fn,chunk=1024*32, 
        vel_model=None, middle_slice=False):
        # middle_slice, only for fast visualization of the middle slice
        D,H,W = self.D,self.H,self.W
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        pts_N = pts_flat.shape[0]
        if t is not None:
            input_t = torch.ones([pts_N, 1])*float(t)
            pts_flat = torch.cat([pts_flat,input_t], dim=-1)


        world_v = self.get_velocity_flat(pts_flat,batchify_fn,chunk,vel_model)
        reso_scale = [self.W*deltaT,self.H*deltaT,self.D*deltaT]
        target_v = vel_world2smoke(world_v, self.s_w2s, self.s_scale, reso_scale)

        if middle_slice:
            _n = 1 if self.middleView=="mid" else 3
            _yzV, _zxV, _xyV = torch.split(target_v, [D*H*_n,D*W*_n,H*W*_n], dim=0)
            mixV = self.__pad_slice_to_volume(_yzV, _n, 0) + self.__pad_slice_to_volume(_zxV, _n, 1) +self.__pad_slice_to_volume(_xyV, _n, 2)
            target_v = mixV / self.npMaskXYZ
        else:
            target_v = target_v.view(D,H,W,3) 
        
        return target_v

    def save_voxel_den_npz(self,den_path,t,use_viewdirs=False,network_query_fn=None, network_fn=None,chunk=1024*32,save_npz=True,save_jpg=False, jpg_mix=True, noStatic=False):
        voxel_den_list = self.get_voxel_density_list(t,chunk,use_viewdirs,network_query_fn,
            network_fn, middle_slice=not (jpg_mix or save_npz) )
        head_tail = os.path.split(den_path)
        namepre = ["","static_"]
        for voxel_den, npre in zip(voxel_den_list, namepre):
            voxel_den = voxel_den.detach().cpu().numpy()
            if save_jpg:
                jpg_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0]+".jpg")
                imageio.imwrite(jpg_path, den_scalar2rgb(voxel_den, scale=None, is3D=True, logv=False, mix=jpg_mix))
            if save_npz:
                # to save some space
                npz_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0]+".npz")
                voxel_den = np.float16(voxel_den)
                np.savez_compressed(npz_path, vel=voxel_den)
            if noStatic:
                break

    def save_voxel_vel_npz(self,vel_path,deltaT,t,batchify_fn,chunk=1024*32, vel_model=None,save_npz=True,save_jpg=False,save_vort=False):
        vel_scale = 160
        voxel_vel = self.get_voxel_velocity(deltaT,t,batchify_fn,chunk,vel_model,middle_slice=not save_npz).detach().cpu().numpy()
        
        if save_jpg:
            jpg_path = os.path.splitext(vel_path)[0]+".jpg"
            imageio.imwrite(jpg_path, vel_uv2hsv(voxel_vel, scale=vel_scale, is3D=True, logv=False))
        if save_npz:
            if save_vort and save_jpg:
                _, NETw = jacobian3D_np(voxel_vel)
                head_tail = os.path.split(vel_path)
                imageio.imwrite( os.path.join(head_tail[0], "vort"+os.path.splitext(head_tail[1])[0]+".jpg"),
                        vel_uv2hsv(NETw[0],scale=vel_scale*5.0,is3D=True) )
            # to save some space
            voxel_vel = np.float16(voxel_vel)
            np.savez_compressed(vel_path, vel=voxel_vel)

#####################################################################
# Loss Tools (all for torch Tensors)
def fade_in_weight(step, start, duration):
    return min(max((float(step) - start)/duration, 0.0), 1.0)
    
# Ghost Density Loss Tool
def ghost_loss_func(_rgb, bg, _acc, den_penalty = 0.0):
    _bg = bg.detach()
    ghost_mask = torch.mean(torch.square(_rgb - _bg), -1)
    ghost_mask = torch.sigmoid(ghost_mask*-1.0) + den_penalty # (0 to 0.5) + den_penalty
    ghost_alpha = ghost_mask * _acc
    return torch.mean(torch.square(ghost_alpha))

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return torch.mean(torch.square(pred - exact))

# VGG Tool, https://github.com/crowsonkb/style-transfer-pytorch/
class VGGFeatures(nn.Module):
    poolings = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d} #, 'l2': partial(nn.LPPool2d, 2)}
    pooling_scales = {'max': 1., 'average': 2., 'l2': 0.78}

    def __init__(self, layers, pooling='max'):
        super().__init__()
        self.layers = sorted(set(layers))

        # The PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, unlike Simonyan et al.'s original model.
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # The PyTorch pre-trained VGG-19 has different parameters from Simonyan et al.'s original
        # model.
        self.model = torchvision.models.vgg19(pretrained=True).features[:self.layers[-1] + 1]
        self.devices = [torch.device('cpu')] * len(self.model)

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        pool_scale = self.pooling_scales[pooling]
        for i, layer in enumerate(self.model):
            if pooling != 'max' and isinstance(layer, nn.MaxPool2d):
                # Changing the pooling type from max results in the scale of activations
                # changing, so rescale them. Gatys et al. (2015) do not do this.
                self.model[i] = Scale(self.poolings[pooling](2), pool_scale)

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    @staticmethod
    def _get_min_size(layers):
        last_layer = max(layers)
        min_size = 1
        for layer in [4, 9, 18, 27, 36]:
            if last_layer < layer:
                break
            min_size *= 2
        return min_size

    def distribute_layers(self, devices):
        for i, layer in enumerate(self.model):
            if i in devices:
                device = torch.device(devices[i])
            self.model[i] = layer.to(device)
            self.devices[i] = device

    def forward(self, input, layers=None):
        # input shape, b,3,h,w
        layers = self.layers if layers is None else sorted(set(layers))
        h, w = input.shape[2:4]
        min_size = self._get_min_size(layers)
        if min(h, w) < min_size:
            raise ValueError(f'Input is {h}x{w} but must be at least {min_size}x{min_size}')
        feats = {'input': input}
        norm_in = torch.stack([self.normalize(input[_i]) for _i in range(input.shape[0])], dim=0)
        # input = self.normalize(input)
        for i in range(max(layers) + 1):
            norm_in = self.model[i](norm_in.to(self.devices[i]))
            if i in layers:
                feats[i] = norm_in
        return feats

# VGG Loss Tool
class VGGlossTool(object):
    def __init__(self, device, pooling='max'):
        # The default content and style layers in Gatys et al. (2015):
        #   content_layers = [22], 'relu4_2'
        #   style_layers = [1, 6, 11, 20, 29], relu layers: [ 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        # We use [5, 10, 19, 28], conv layers before relu: [ 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.layer_list = [5, 10, 19, 28]        
        self.layer_names = [
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.device = device

        # Build a VGG19 model loaded with pre-trained ImageNet weights
        self.vggmodel = VGGFeatures(self.layer_list, pooling=pooling)
        device_plan = {0: device}
        self.vggmodel.distribute_layers(device_plan)

    def feature_norm(self, feature):
        # feature: b,h,w,c
        feature_len = torch.sqrt(torch.sum(torch.square(feature), dim=-1, keepdim=True)+1e-12)
        norm = feature / feature_len
        return norm

    def cos_sim(self, a,b):
        cos_sim_ab = torch.sum(a*b, dim=-1)
        # cosine similarity, -1~1, 1 best
        cos_sim_ab_score = 1.0 - torch.mean(cos_sim_ab) # 0 ~ 2, 0 best
        return cos_sim_ab_score

    def compute_cos_loss(self, img, ref):
        # input img, ref should be in range of [0,1]
        input_tensor = torch.stack( [ref, img], dim=0 )
        
        input_tensor = input_tensor.permute((0, 3, 1, 2))
        # print(input_tensor.shape)
        _feats = self.vggmodel(input_tensor, layers=self.layer_list)

        # Initialize the loss
        loss = []
        # Add loss
        for layer_i, layer_name in zip (self.layer_list, self.layer_names):
            cur_feature = _feats[layer_i]
            reference_features = self.feature_norm(cur_feature[0, ...])
            img_features = self.feature_norm(cur_feature[1, ...])

            feature_metric = self.cos_sim(reference_features, img_features)
            loss += [feature_metric]
        return loss


