# Physics Informed Neural Fields for Smoke Reconstruction with Sparse Data
This is the pytorch implementation of the paper, "Physics Informed Neural Fields for Smoke Reconstruction with Sparse Data" (SIGGRAPH 2022, TOG journal paper), from M. Chu, L. Liu, Q. Zheng, E. Franz, H.P. Seidel, C. Theobalt, R. Zayer. 
The [project page](https://rachelcmy.github.io/pinf_smoke/) contains the paper and other materials.  
![](./imgs/teaser.jpg)

## Installation

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ### Dependencies
  - Python 3.7
  - PyTorch 1.6 (The code is tested on GPU with CUDA 10.2)
  - matplotlib
  - numpy
  - imageio
  - imageio-ffmpeg
  - configargparse  
</details>

### Install Python 3.7 and PyTorch 1.6 (The code is tested on GPU with CUDA 10.2)

As a starting point, a new conda environment ([miniconda](https://docs.conda.io/en/latest/miniconda.html)) is recommended.  
```
# clone the repo and open the base directory
git clone https://github.com/RachelCmy/PINF-SmokeRecon.git  
cd PINF-SmokeRecon

# build environment with python 3.7
conda create -n pinf_ENV python=3.7
conda activate pinf_ENV # all following operations are using this environment.

# if ffmpeg is not installed (test by ffmpeg -version)
conda install -c conda-forge ffmpeg

# install PyTorch 1.6.0
conda install pytorch==1.6.0 torchvision==0.7.0 torchaudio cudatoolkit=10.2 -c pytorch
```

### Install Other Packages
```
# install required packages
pip install -r requirements.txt

# simple installation test
python simpletest.py
```

## Scenes
<details>
  <summary> Scene Summary (click to expand) </summary>
  
  ### Scenes
  - Real Captures (provided by ScalarFlow and GlobalTrans)
  - Synthetic Scene: Sphere
  - Synthetic Scene: Game
</details>

Use the following commands to download test scenes:
```
bash download_example_data.sh
```
Under the "data/", there will be there sub-folders, "ScalarReal", "Sphere", and "Game".

[Mantaflow](http://www.mantaflow.com/) and [Blender](https://www.blender.org/) are used to generate the input videos of synthetic scenes. The source code and scene files used to generate the videos are not included in the repo, but there are some instructions and example scripts (used for the game scene) here: [Synthetic_Data_ReadMe](./data/Game/scripts/)


## Run the Reconstruction Code

```
# To Reconstruct the Real ScalarFlow Scene:
python run_pinf.py --config configs/scalarReal.txt
# To Reconstruct the Synthetic Hybrid Scene with a Sphere Obstacle:
python run_pinf.py --config configs/sphere.txt
# To Reconstruct the Game Scene:
python run_pinf.py --config configs/game.txt
```
After training for 50k iterations (~4 hours on a single NVIDIA Quadro RTX 8000 GPU), you can find the following video at `logs/xxxxx/xxxxx_spiral_050000_rgb.mp4` and `logs/xxxxx/xxxxx_volume_050000_velrgb.mp4`, which render the radiance and visualize the velocity fields.

<!-- <div>like these (some early results):
<table width="100%" align="center" style="vertical-align:top">
  <tr style="text-align: center;">
    <td width="40%">
        <img width="100%"src="https://people.mpi-inf.mpg.de/~mchu/projects/PI-NeRF/content/sphere_test_spiral_050000_rgb.gif" >
    </td>
    <td width="60%">
    <img width="100%" src="https://people.mpi-inf.mpg.de/~mchu/projects/PI-NeRF/content/sphere_test_volume_050000_velrgb.gif" >
    </td>
  </tr>
</table>    
</div> -->

The full training takes 200k to 600k iterations (around 1 day or longer on a single NVIDIA Quadro RTX 8000 GPU). The results of the papar ( given in the [supplemental materials](https://rachelcmy.github.io/pinf_smoke/ClickMe.html) ) are generated with code based on Tensorflow. We publish the PyTorch version which is around 1-1.5 times faster.

__Static NeRF Support:__ Static NeRF scenes can be reconstructed in the same way as in [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch)  

```
# Download data for two NeRF datasets: `lego` and `fern`
bash download_nerf_data.sh
# To train a low-res `lego` NeRF:
python run_nerf.py --config nerf_configs/lego.txt
# To train a low-res `fern` NeRF:
python run_nerf.py --config nerf_configs/fern.txt
```

__GPU Memory__: The code uses one single GPU. Code modification is necessary to support multiple GPUs. Taking the Sphere scene as an example, it uses 16G memory on Quadro RTX 8000, which has 48G memory in total. It is possible to train coarse models with sparsely sampled points using 12G GPU memory by setting ``N_samples = 32`` and ``vol_output_W = 96``.

## BibTex citation

When using this code, please cite our preprint ``Chu et al.: Physics Informed Neural Fields for Smoke Reconstruction with Sparse Data`` as well as the following works on which it builds:
```
@article{chu2022physics
 author = {Chu, Mengyu and Liu, Lingjie and Zheng, Quan and Franz, Erik and Seidel, Hans-Peter and Theobalt, Christian and Zayer, Rhaleb},
 title = {Physics Informed Neural Fields for Smoke Reconstruction with Sparse Data},
 journal = {ACM Transactions on Graphics},
 month = {aug},
 number = {4},
 articleno = {119},
 pages = {119:1-119:14},
 publisher = {Association for Computing Machinery},
 address = {New York, NY, USA},
 volume = {41},
 year = {2022},
 issue_date = {July 2022},
}
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
@inproceedings{mildenhall2020nerf,
 title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
 author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
 year={2020},
 booktitle={ECCV},
}
```

## License

This code builds on the [PyTorch port by Yen-Chen Lin](https://github.com/yenchenlin/nerf-pytorch) of the [original NeRF code](https://github.com/bmild/nerf). We thank all of them for releasing their code.

We release this code under an MIT license as well. You can find all licenses in the file `LICENSE`.