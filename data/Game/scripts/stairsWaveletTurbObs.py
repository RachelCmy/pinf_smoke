################################################################################
# mantaflow scene file
# check out http://mantaflow.com/quickstart.html 
# for mantaflow installation and quick guide to run scene files.
################################################################################
# Complexer smoke simulation with wavlet turbulence plus
# - obstacle handling
# - and uv coordinates
# #############################################################################

from manta import *
import os, shutil, math, sys, cv2 as cv
import numpy as np

# how much to upres the XL sim?
# set to zero to disable the second one completely
upres = 4
 
# overall wavelet noise strength
wltStrength = 0.3 

# how many grids of uv coordinates to use (more than 2 usually dont pay off here)
uvs = 1

# and how many octaves of wavelet turbulence? could be set manually, but ideally given by upres factor (round)
octaves = 0
if(upres>0):
    octaves = int( math.log(upres)/ math.log(2.0) + 0.5 )

# simulation resolution 
dim = 3
res = 128
bWidth=1
gs = vec3(res,int(res*0.8),int(res*0.85))
if (dim==2): gs.z = 1 

# setup low-res sim
sm = Solver(name='main', gridSize = gs, dim=dim)
sm.timestep = 1.0
# to do larger timesteps, without adaptive time steps, we need to set the length of a frame (1 is default)
sm.frameLength = sm.timestep 
timings = Timings()

# note - world space velocity, convert to grid space later


# inflow noise field
noise = NoiseField( parent=sm, fixedSeed=265, loadFromFile=True)
noise.posScale = vec3(20) # note, this is normalized to the grid size...
noise.clamp = True
noise.clampNeg = 0
noise.clampPos = 3
noise.valScale = 1
noise.valOffset = 0.075
noise.timeAnim = 0.3

velInflow = vec3(0.01, 0, -0.01)
source_pos = vec3(0.42,0.36,0.92)
# helper objects: inflow region, and obstacle
source    = Cylinder( parent=sm, center=gs*source_pos, radius=res*0.05, z=gs*vec3(0.061, 0, -0.081))
sourceVel = Cylinder( parent=sm, center=gs*source_pos, radius=res*0.05 , z=gs*vec3(0.062 , 0, -0.1))

velInflow2 = vec3(0.002, 0, -0.014)
source_pos2 = vec3(0.32,0.36,0.95)
# helper objects: inflow region, and obstacle
source2    = Cylinder( parent=sm, center=gs*source_pos2, radius=res*0.05, z=gs*vec3(0.02, 0, -0.11))
sourceVel2 = Cylinder( parent=sm, center=gs*source_pos2, radius=res*0.05 , z=gs*vec3(0.03 , 0, -0.14))

# larger solver, recompute sizes...
# init lower res solver & grids
if(upres>0):
    xl_gs = vec3(upres*gs.x,upres*gs.y,upres*gs.z)
    if (dim==2): xl_gs.z = 1  # 2D
    xl = Solver(name='larger', gridSize = xl_gs, dim=dim)
    xl.timestep    = sm.timestep 
    xl.frameLength = xl.timestep 

    xl_flags   = xl.create(FlagGrid)
    xl_vel     = xl.create(MACGrid)
    xl_density = xl.create(RealGrid)

    stillphi= xl.create(LevelsetGrid)
    xl_flags.initDomain(outflow="xXYzZ", phiWalls=stillphi, boundaryWidth=bWidth*upres)
    xl_flags.fillGrid()

    xl_source = Cylinder( parent=xl, center=xl_gs*source_pos, radius=xl_gs.x*0.05, z=xl_gs*vec3(0.061, 0, -0.081))
    xl_source1 = Cylinder( parent=xl, center=xl_gs*source_pos2, radius=xl_gs.x*0.05, z=xl_gs*vec3(0.02, 0, -0.11))
    # xl_obs    = Sphere(   parent=xl, center=xl_gs*vec3(0.5,0.5,0.5), radius=xl_gs.x*0.15)
    # xl_obs.applyToGrid(grid=xl_flags, value=FlagObstacle)

    xl_noise = NoiseField( parent=xl, fixedSeed=265, loadFromFile=True)
    xl_noise.posScale = noise.posScale
    xl_noise.clamp    = noise.clamp
    xl_noise.clampNeg = noise.clampNeg
    xl_noise.clampPos = noise.clampPos
    xl_noise.valScale = noise.valScale
    xl_noise.valOffset = noise.valOffset
    xl_noise.timeAnim  = noise.timeAnim * upres

flags = sm.create(FlagGrid)
lowphi = sm.create(LevelsetGrid)
flags.initDomain(outflow="xXYzZ", phiWalls=lowphi, boundaryWidth=bWidth)
flags.fillGrid()
setOpenBound(flags, bWidth,'xXYzZ', FlagOutflow|FlagEmpty) 
# obs.applyToGrid(grid=flags, value=FlagObstacle)

obj_path = "./nerfdata/tower/manta/"
mesh_list = ["Stairs.obj", "StairsObj2.obj"]

def loadMesh(s, gs, mpath):
    _mesh = s.create(Mesh)
    _mesh.load(mpath)
    _mesh.scale(vec3(0.3*gs.x))
    _mesh.offset(vec3(0.5, 0.0, 0.65)*gs)  # 0.73 - 0.7 0.65 0.63 0.62
    return _mesh

if False: # load obstacles, and generate sdf
    viewgrid = sm.create(RealGrid)
    xl_viewgrid = xl.create(RealGrid)
    interpphi = xl.create(LevelsetGrid)

    def saveSDFunippm(outPhi, sdfsig, viewGrid, _path):
        copyLevelsetToReal (source=outPhi , target=viewGrid)
        viewGrid.save(_path+".uni")
        viewGrid.multConst(-1.0)
        viewGrid.addConst(sdfsig)
        projectPpmFull( viewGrid, _path+".ppm", 0, 4.0/_sig )


    def calsdf(inMesh, outPhi, sdfsig, viewGrid, _path, save=False):
        inMesh.computeLevelset(outPhi, sdfsig)        
        if save:
            saveSDFunippm(outPhi, sdfsig, viewGrid, _path)

    _sig = 3 # 2.4
    for mesh_path in mesh_list:
        _smPhi = sm.create(LevelsetGrid)
        _mesh = loadMesh(sm, gs, obj_path + mesh_path)
        
        calsdf(_mesh, _smPhi, _sig, viewgrid, obj_path + mesh_path[:-4])
        interpolateGrid(source=_smPhi , target=interpphi)
        interpphi.multConst(upres)
        lowphi.join(_smPhi)
        stillphi.join(interpphi) 
        _mesh.applyMeshToGridWithSDF( flags, _smPhi, value=FlagObstacle)
        _mesh.applyMeshToGridWithSDF(xl_flags, interpphi, value=FlagObstacle)

    saveSDFunippm(stillphi, _sig, xl_viewgrid, obj_path+'stair_fine')
    saveSDFunippm(lowphi, _sig, viewgrid, obj_path+'stair_coarse')
    xl_flags.save(obj_path+'stair_fine_flag.uni')
    flags.save(obj_path+'stair_coarse_flag.uni')
else: # load previously generated sdf
    for mesh_path in mesh_list: # only for visualization
        _smPhi = sm.create(LevelsetGrid)
        _mesh = loadMesh(sm, gs, obj_path + mesh_path)

    stillphi.load(obj_path + 'stair_fine.uni')
    lowphi.load(obj_path + 'stair_coarse.uni')
    xl_flags.load(obj_path+'stair_fine_flag.uni')
    flags.load(obj_path+'stair_coarse_flag.uni')


# create the array of uv grids
uv = []
for i in range(uvs):
    uvGrid = sm.create(VecGrid)
    uv.append(uvGrid)
    resetUvGrid( uv[i] )

vel       = sm.create(MACGrid) 
density   = sm.create(RealGrid)
pressure  = sm.create(RealGrid)
energy    = sm.create(RealGrid)
tempFlag  = sm.create(FlagGrid)

# wavelet turbulence noise field
xl_wltnoise = NoiseField( parent=xl, loadFromFile=True)
# scale according to lowres sim , smaller numbers mean larger vortices
# note - this noise is parented to xl solver, thus will automatically rescale
xl_wltnoise.posScale = vec3( int(1.0*gs.x) ) * 0.5
xl_wltnoise.timeAnim = 0.1


ar_sm = np.zeros([1,int(gs.z), int(gs.y), int(gs.x),1], dtype=np.float32)
ar_xl = np.zeros([1,int(xl_gs.z), int(xl_gs.y), int(xl_gs.x),1], dtype=np.float32)

arV_sm = np.zeros([1,int(gs.z), int(gs.y), int(gs.x),3], dtype=np.float32)
arV_xl = np.zeros([1,int(xl_gs.z), int(xl_gs.y), int(xl_gs.x),3], dtype=np.float32)
# setup user interface
if (GUI):
    gui = Gui()
    sliderStr = gui.addControl(Slider, text='turb. strength', val=wltStrength, min=0, max=2)
    gui.show()
    # gui.pause()
else:
    exit()
#printBuildInfo() 

def save_fig(den, denArray, vel, image_dir, t, is2D):
    print("save_fig", image_dir, t, flush=True)

    saveppm, savenpz = True, True
    denArray = np.clip(denArray, 0, 10)
    copyArrayToGridReal(source=denArray, target=den )

    if is2D:
        if saveppm:
            projectPpmFull( den, image_dir+'den_%04d.ppm' % (t), 0, 1.0 )
        # cv.imwrite(image_dir+'vel_%04d.png' % (t), vel_uv2hsv(vel[0])[::-1,:,::-1])
        # _, NETw = jacobian2D_np(vel)
        # cv.imwrite(image_dir+'vor_%04d.png' % (t), vor_rgb(NETw[0])[::-1,:,::-1])
        if savenpz:
            np.savez_compressed( image_dir+'den_%04d.npz' % (t), denArray )
            np.savez_compressed( image_dir+'vel_%04d.npz' % (t), vel )
    else:
        if saveppm:
            projectPpmFull( den, image_dir+'den_%04d.ppm' % (t), 0, 4.0 )
        # cv.imwrite(image_dir+'vel_%04d.png' % (t), 
        #     vel_uv2hsv(vel[0],scale=256, is3D=True, logv=False)[::-1,:,::-1])
        if savenpz:
            tosave = np.float16(denArray)
            # diff = array - np.float32(tosave)
            np.savez_compressed( image_dir+'den_%04d.f16.npz' % (t) , tosave )
            tosave = np.float16(vel)
            np.savez_compressed( image_dir+'vel_%04d.f16.npz' % (t), tosave )
        

# upres = 0
# main loop
for t in range(200):
    mantaMsg('\nFrame %i, simulation time %f' % (sm.frame, sm.timeTotal))

    if (GUI):
        wltStrength = sliderStr.get()

    advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, orderSpace=2, strength=0.6, clampMode=2, openBounds=True, boundaryWidth=bWidth)
    advectSemiLagrange(flags=flags, vel=vel, grid=vel,      order=2, clampMode=2, openBounds=True, boundaryWidth=bWidth )

    for i in range(uvs):
        advectSemiLagrange(flags=flags, vel=vel, grid=uv[i], order=2, orderSpace=2, strength=0.6, clampMode=2, openBounds=True, boundaryWidth=bWidth)
        # now we have to update the weights of the different uv channels
        # note: we have a timestep of 1.5 in this setup! so the value of 16.5 means reset every 11 steps
        updateUvWeight( resetTime=16.5 , index=i, numUvs=uvs, uv=uv[i] ); 
        # also note, we have to update the weight after the advection 
        # as it is stored at uv[i](0,0,0) , the advection overwrites this...
        
    applyInflow=False
    if (sm.timeTotal>=0 and sm.timeTotal<120.):
        vorticityConfinement( vel=vel, flags=flags, strength=0.05 )

        densityInflow( flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5 )
        sourceVel.applyToGrid( grid=vel , value=(velInflow*float(res)) )

        densityInflow( flags=flags, density=density, noise=noise, shape=source2, scale=1, sigma=0.5 )
        sourceVel2.applyToGrid( grid=vel , value=(velInflow2*float(res)) )
        applyInflow=True
        
    
    setWallBcs(flags=flags, vel=vel)    
    addBuoyancy(density=density, vel=vel, gravity=vec3(0,-2e-4,0), flags=flags)
    solvePressure(flags=flags, vel=vel, pressure=pressure , cgMaxIterFac=99, cgAccuracy=1e-05, zeroPressureFixing=True, preconditioner = PcMGStatic)
    setWallBcs(flags=flags, vel=vel)
    
    # determine weighting
    computeEnergy(flags=flags, vel=vel, energy=energy)

    # mark outer obstacle region by extrapolating flags for 2 layers
    tempFlag.copyFrom(flags)
    extrapolateSimpleFlags( flags=flags, val=tempFlag, distance=2, flagFrom=FlagObstacle, flagTo=FlagFluid );
    # now extrapolate energy weights into obstacles to fix boundary layer
    extrapolateSimpleFlags( flags=tempFlag, val=energy, distance=6, flagFrom=FlagFluid, flagTo=FlagObstacle ); 
    computeWaveletCoeffs(energy)

    #density.save('densitySm_%04d.vol' % t)
    copyGridToArrayReal(source=density, target=ar_sm )
    copyGridToArrayMAC(source=vel, target=arV_sm ) 

    save_fig(density, ar_sm, arV_sm, obj_path+"stair_den/sm_", t, False)
    sm.step()
    
    # xl ...
    if(upres>0):
        interpolateMACGrid( source=vel, target=xl_vel )
        
        # add all necessary octaves
        sStr = 1.0 * wltStrength 
        sPos = 2.0 
        for o in range(octaves):
            # add wavelet noise for each grid of uv coordinates 
            #xl_vel.clear() # debug , show only noise eval 
            for i in range(uvs):
                uvWeight = getUvWeight(uv[i]) 
                applyNoiseVec3( flags=xl_flags, target=xl_vel, noise=xl_wltnoise, scale=sStr * uvWeight, scaleSpatial=sPos , 
                    weight=energy, uv=uv[i] )
            #mantaMsg( "Octave "+str(o)+", ss="+str(sStr)+" sp="+str(sPos)+" uvs="+str(uvs) ) # debug output 

            # update octave parameters for next iteration
            sStr *= 0.06 # magic kolmogorov factor
            sPos *= 2.0 
        
        # now advect
        for substep in range(upres): 
            advectSemiLagrange(flags=xl_flags, vel=xl_vel, grid=xl_density, order=2, orderSpace=2, strength=0.6, clampMode=2, openBounds=True, boundaryWidth=bWidth)
        # manually recreate inflow
        if (applyInflow):
            densityInflow( flags=xl_flags, density=xl_density, noise=xl_noise, shape=xl_source, scale=1, sigma=0.5 )
            densityInflow( flags=xl_flags, density=xl_density, noise=xl_noise, shape=xl_source2, scale=1, sigma=0.5 )

        copyGridToArrayReal(source=xl_density, target=ar_xl )
        copyGridToArrayMAC(source=xl_vel, target=arV_xl ) 
        save_fig(xl_density, ar_xl, arV_xl, obj_path+"stair_den/xl_", t, False)
        xl.step()    

    #timings.display()
    # small and xl grid update done
    #gui.screenshot( 'wltObs_%04d.png' % t );

