#!/bin/bash

# Check if Python version is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <ENV_NAME> <PYTHON_VERSION>"
    exit 1
fi

ENV_NAME=$1
PYTHON_VERSION=$2

conda create -n $ENV_NAME python=$PYTHON_VERSION -y
conda activate $ENV_NAME
# Install PyTorch with CUDA support
# For CUDA 12.4, use the following command
# CUDA 12.4
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "pybind11[global]"

# git clone 

git clone git@github.com:zyxkv/EzSim.git
cd EzSim && \
git submodule update --init --recursive

cd EzSim/ezsim/ext/LuisaRender && \
git submodule update --init --recursive && \
mkdir -p build && \
cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D PYTHON_VERSIONS=$PYTHON_VERSION \
    -D LUISA_COMPUTE_DOWNLOAD_NVCOMP=ON \
    -D LUISA_COMPUTE_DOWNLOAD_OIDN=ON \
    -D LUISA_COMPUTE_ENABLE_GUI=OFF \
    -D LUISA_COMPUTE_ENABLE_CUDA=ON \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") && \
cmake --build build -j $(nproc)



pip install --no-cache-dir open3d
git clone https://github.com/zyxkv/EzSim.git && \
    cd EzSim && \
    pip install . && \
    pip install --no-cache-dir PyOpenGL==3.1.5

# -------------------- Surface Reconstruction --------------------
# Set the LD_LIBRARY_PATH directly in the environment
COPY --from=builder /workspace/EzSim/ezsim/ext/ParticleMesher/ParticleMesherPy /opt/conda/lib/python3.11/site-packages/ezsim/ext/ParticleMesher/ParticleMesherPy
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/ezsim/ext/ParticleMesher/ParticleMesherPy:$LD_LIBRARY_PATH

# --------------------- Ray Tracing Renderer ---------------------
# Copy LuisaRender build artifacts from the builder stage
COPY --from=builder /workspace/EzSim/ezsim/ext/LuisaRender/build/bin /opt/conda/lib/python3.11/site-packages/ezsim/ext/LuisaRender/build/bin
# fix GLIBCXX_3.4.30 not found
RUN cd /opt/conda/lib && \
    mv libstdc++.so.6 libstdc++.so.6.old && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6