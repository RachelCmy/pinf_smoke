import sys, os
import numpy as np
import pyopenvdb as vdb

target_path = "./mydata/blendersyn/blenderVDB/fluid_%04d.vdb"
manta_path = "./mydata/blendersyn/synthData/density_%06d.npz"
vscale = 0.4909/100.0

dengrid = vdb.FloatGrid()
dengrid.name = 'density'
dengrid.gridClass = vdb.GridClass.FOG_VOLUME
dengrid.transform = vdb.createLinearTransform(voxelSize=vscale)

meta_dict = {"active_fields":0, "blender/smoke/active_fields": 5}
for framei in range(415, 560): # [100]
    den_file = manta_path%framei
    tar_denfile = target_path%framei
    dengrid.clear()

    manta_grid = np.load(den_file)
    if 'data' in manta_grid:
        manta_grid = manta_grid['data']
    elif "arr_0"in manta_grid:
        manta_grid = manta_grid["arr_0"]
    manta_grid = np.float32(np.squeeze(manta_grid))
    print(manta_grid.shape)
    
    dengrid.copyFromArray (manta_grid)

    tosave = [dengrid]
    vdb.write(tar_denfile, grids=tosave, metadata=meta_dict)