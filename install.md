# Prerequisites

This repo has been tested on:

- Linux (tested on Ubuntu 18.04/20.04 LTS)
- Python=3.8
- PyTorch=1.11.0
- CUDA=11.3
- mmcv=1.5.3
- mmdetection=2.25.3
- mmsegmentation=0.26.0
- mmdetection3d=1.0.0rc6

An example script for installing the python dependencies under CUDA 11.3:
```
# Export the PATH of CUDA toolkit
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

# Create conda environment
conda create -y -n monoxiver python=3.8
conda activate monoxiver

# Install pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install external libs
mkdir external
cd external

# Install mmcv
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout 30f305810702f47525a28e6a58d52414ecb79d0f # v1.5.3
MMCV_WITH_OPS=1 pip install -e .
cd ..

# Install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout e71b499608e9c3ccd4211e7c815fa20eeedf18a2 # v2.25.3
pip install -r requirements/build.txt
pip install -v -e .
cd ..

# install mmsegmentation
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout 17056b636f61b6887d72f492178b0399a46ab4d8 # v0.26.0
pip install -e .
cd ..

# install mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout 47285b3f1e9dba358e98fcd12e523cfd0769c876 # v1.0.0rc6
pip install -e .
cd ..

pip install timm==0.6.7  # custom timm version
pip install numba==0.56.4
pip install yapf==0.40.1
cd ..

# install ivmclx package
pip install -v -e .
```