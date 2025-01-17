# https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html#variable:CMAKE_%3CLANG%3E_FLAGS
export CUDAFLAGS := "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 --expt-relaxed-constexpr"
export CUDA_ARCHITECTURES := "native"
project_dir := justfile_directory()
libtorch_url := "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip"
TORCH_CUDA_ARCH_LIST := "8.0 8.6"
NINJA_MAX_JOBS := num_cpus()
CMAKE_MAX_JOBS := num_cpus()
CU_BUILD_TARGETS := ""
PYTEST_PATTERN := "test_.*"

clean:
    # python
    if [ -d {{project_dir}}/dist ]; then rm -r {{project_dir}}/dist; fi
    if [ -d {{project_dir}}/src/mase_cuda.egg-info ]; then rm -r {{project_dir}}/src/mase_cuda.egg-info; fi
    # all
    if [ -d {{project_dir}}/build ]; then rm -r {{project_dir}}/build; fi

# ==================== C++ ======================
[private]
cmake-build:
    if [ -z {{CU_BUILD_TARGETS}} ]; \
        then cmake --build {{project_dir}}/build -j {{CMAKE_MAX_JOBS}} ; \
    else \
        cmake --build {{project_dir}}/build --target {{CU_BUILD_TARGETS}} -j {{CMAKE_MAX_JOBS}} ; \
    fi

build-cu-test: clean && cmake-build
    echo $(which cmake)
    cmake -D BUILD_TESTING=ON -D CUDA_ARCHITECTURES={{CUDA_ARCHITECTURES}} -B build -S .

build-cu-test-debug: clean && cmake-build
    cmake -D BUILD_TESTING=ON -D CUDA_ARCHITECTURES={{CUDA_ARCHITECTURES}} -D NVCCGDB=ON -B build -S .

build-cu-profile: clean && cmake-build
    cmake -D BUILD_PROFILING=ON -D CUDA_ARCHITECTURES={{CUDA_ARCHITECTURES}} -B build -S .

# ==================== Python ====================
build-py: clean
    TORCH_CUDA_ARCH_LIST="{{TORCH_CUDA_ARCH_LIST}}" MAX_JOBS={{NINJA_MAX_JOBS}} tox -e build

test-py-fast:
    TORCH_CUDA_ARCH_LIST="{{TORCH_CUDA_ARCH_LIST}}" MAX_JOBS={{NINJA_MAX_JOBS}} tox r -e py311 -- -v --log-cli-level INFO -m "not slow" --durations=0

test-py-slow:
    TORCH_CUDA_ARCH_LIST="{{TORCH_CUDA_ARCH_LIST}}" MAX_JOBS={{NINJA_MAX_JOBS}} tox r -e py311 -- -v --log-cli-level INFO -m "slow" --durations=0

test-py:
    TORCH_CUDA_ARCH_LIST="{{TORCH_CUDA_ARCH_LIST}}" MAX_JOBS={{NINJA_MAX_JOBS}} tox r -e py311 -- -v --log-cli-level INFO

test-py-pattern:
    echo "Function pattern: {{PYTEST_PATTERN}}"
    TORCH_CUDA_ARCH_LIST="{{TORCH_CUDA_ARCH_LIST}}" MAX_JOBS={{NINJA_MAX_JOBS}} tox r -e py311 -- -v --log-cli-level INFO -k "{{PYTEST_PATTERN}}"

# build, test, and package
tox:
    TORCH_CUDA_ARCH_LIST="{{TORCH_CUDA_ARCH_LIST}}" MAX_JOBS={{NINJA_MAX_JOBS}} tox

# ==================== Utils ====================
download-libtorch-if-not-exists:
    if [ ! -d {{project_dir}}/submodules/libtorch ]; then curl -L {{libtorch_url}} -o {{project_dir}}/submodules/libtorch.zip; unzip {{project_dir}}/submodules/libtorch.zip -d {{project_dir}}/submodules; else echo "libtorch already exists"; fi