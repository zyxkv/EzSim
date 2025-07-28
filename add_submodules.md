# 0. 清理submodules和已经存在的索引
git add ezsim/ext/LuisaRender ezsim/ext/ParticleMesher
git rm --cached ezsim/ext/LuisaRender ezsim/ext/ParticleMesher 2>/dev/null || true
git commit -m "Remove old submodules before reconfiguration"
rm -rf .git/modules/ezsim/ext/LuisaRender .git/modules/ezsim/ext/ParticleMesher
# 1. 重新添加如下两个submodules
git submodule add https://github.com/Alif-01/LuisaRender.git ezsim/ext/LuisaRender
git submodule add https://github.com/ACMLCZH/ParticleMesher.git ezsim/ext/ParticleMesher

# 2. 进入每个子模块目录，检查远程仓库配置
cd ezsim/ext/LuisaRender
git remote -v
cd ../../../

cd ezsim/ext/ParticleMesher  
git remote -v
cd ../../../

# 3. 重新初始化子模块（注意子模块中也有submodules，需要全量git）
git submodule init
git submodule sync --recursive

# 4. 尝试编译 LuisaRender 测试是否全量git到正确的submodules
cd ezsim/ext/LuisaRender && \
git submodule update --init --recursive && \
mkdir -p build && \
cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D PYTHON_VERSIONS=3.11 \
    -D LUISA_COMPUTE_DOWNLOAD_NVCOMP=ON \
    -D LUISA_COMPUTE_DOWNLOAD_OIDN=ON \
    -D LUISA_COMPUTE_ENABLE_GUI=OFF \
    -D LUISA_COMPUTE_ENABLE_CUDA=ON \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") && \
cmake --build build -j $(nproc)


# reinit
git add ezsim/ext/LuisaRender ezsim/ext/ParticleMesher
git rm --cached ezsim/ext/LuisaRender ezsim/ext/ParticleMesher 2>/dev/null || true
git commit -m "Remove old submodules before reconfiguration"
rm -rf .git/modules/ezsim/ext/LuisaRender .git/modules/ezsim/ext/ParticleMesher
rm -rf ezsim/ext/LuisaRender ezsim/ext/ParticleMesher
# add
git submodule add https://github.com/Alif-01/LuisaRender.git ezsim/ext/LuisaRender
git submodule add https://github.com/ACMLCZH/ParticleMesher.git ezsim/ext/ParticleMesher
cd ezsim/ext/LuisaRender && git remote -v && cd ../../..
cd ezsim/ext/ParticleMesher && git remote -v && cd ../../..
# 重新初始化子模块
git submodule init
git submodule sync --recursive
cd ezsim/ext/LuisaRender && git submodule update --init --recursive

cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D PYTHON_VERSIONS=3.11 -D LUISA_COMPUTE_DOWNLOAD_NVCOMP=ON -D LUISA_COMPUTE_DOWNLOAD_OIDN=ON -D LUISA_COMPUTE_ENABLE_GUI=OFF -D LUISA_COMPUTE_ENABLE_CUDA=ON -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") -DRust_FIND_QUIETLY=ON -DRust_RESOLVE_RUSTUP_TOOLCHAINS=OFF