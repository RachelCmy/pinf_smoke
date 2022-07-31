## Very brief instructions for synthetic scene generation
Here are some very brief instructions for the synthetic scene generation.  
To work with a new synthetic scenes, it is necessary to write the the simulation script of the scene (Step 1), use blender to render the scene (Step 2), write code to properly output camera poses (Step 3), and modify code (like in [repo_dir/load_pinf.py](../../../load_pinf.py)) to load the image sequences with camera poses. 

### Step 1. We use [Mantaflow](http://www.mantaflow.com/) to simulate synthetic scenes.
An example script (used for the Game scene) can be found here: [./stairsWaveletTurbObs.py](./stairsWaveletTurbObs.py)  
After [Mantaflow installation](http://mantaflow.com/install.html), this scene file can be run with something like:
```
./build/manta ./stairsWaveletTurbObs.py
```
It will save 3D density sequences as npz files (np.float16 is used to save space).

### Step 2. Convert npz files to vdb files for rendering with [Blender](https://www.blender.org/)
After [OpenVDB installation](https://academysoftwarefoundation.github.io/openvdb/index.html), vdb files can be obtained using the following command.
```
python ./manta2vdb.py
```

### Step 3. Use Blender to rendering synthetic images and output camera poses
The vdb sequences can be loaded in Blender. Arbitrary rendering parameters and lighting conditions can be applied. We modify [this code](https://github.com/bmild/nerf/files/4410324/360_view.py.zip) (a zip file provided [here](https://github.com/bmild/nerf/issues/78)) to render images and output camera poses. 


