#!/bin/bash

# Check if Python version is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <ENV_NAME> "
    exit 1
fi

ENV_NAME=$1

conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME
# Install PyTorch with CUDA support
# For CUDA 12.4, use the following command
# CUDA 12.4
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
git clone git@github.com:zyxkv/EzSim.git
# Install dependencies
cd EzSim && \
git submodule update --init --recursive
pip install -e ".[render]"


# ParticleMesherPy (Surface Reconstruction)
echo "export LD_LIBRARY_PATH=${PWD}/ezsim/ext/ParticleMesher/ParticleMesherPy:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
# LuisaRender
cd ezsim/ext/LuisaRender
# make sure git submodule status
# ac034080af1f537f4243bfa4914dd27adbca3589 ezsim/ext/LuisaRender (heads/my_next_compute)
# 006bb10c78f9688a26ed7f7d39da24a3be8c6f8a ezsim/ext/ParticleMesher (heads/main)
# change src/compute/src/ext/corrosion/cmake/FindRust.cmake Line 187 to nextline
# set(Rust_RESOLVE_RUSTUP_TOOLCHAINS OFF CACHE BOOL ${_RESOLVE_RUSTUP_TOOLCHAINS_DESC})

    
cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D PYTHON_VERSIONS=3.10 \
    -D LUISA_COMPUTE_DOWNLOAD_NVCOMP=ON \
    -D LUISA_COMPUTE_ENABLE_GUI=OFF \
    -D LUISA_RENDER_BUILD_TESTS=OFF
    

    # next lines always make aompile stuck at ***-osl (do not use )
    # -D LUISA_COMPUTE_ENABLE_CUDA=ON \
    # -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
    # -D CUDAToolkit_ROOT=/usr/local/cuda-12.4

    
cmake --build build -j $(nproc)
# pymeshlib Qt5 lib
# add the following line to ~/.bashrc
# export LD_LIBRARY_PATH=/home/user/<conda_root>/envs/<ENV_NAME>/lib/python3.10/site-packages/pymeshlab/lib:$LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=/home/user/<conda_root>/envs/<ENV_NAME>/lib/python3.10/site-packages/pymeshlab/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

# libstdc++.so.6 fix
cd $CONDA_PREFIX/lib
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6

# usd
pip install -e .[usd]
# Omniverse kit is used for USD material baking. Only available for Python 3.10 and GPU backend now.
# If USD baking is disabled, Genesis only parses materials of "UsdPreviewSurface".
pip install --extra-index-url https://pypi.nvidia.com/ omniverse-kit
# To use USD baking, you should set environment variable `OMNI_KIT_ACCEPT_EULA` to accept the EULA.
# This is a one-time operation, if accepted, it will not ask again.
export OMNI_KIT_ACCEPT_EULA=yes