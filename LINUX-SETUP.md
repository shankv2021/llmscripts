## Steps to install python, vllm, postgres environment in Redhat Enterprise linux env

> Before any of this check CUDA version, GCC version of your target linux server

### OS level installs (GCC, CMAKE and build tools)
```
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y gcc gcc-c++ make git wget curl which cmake
```

### CUDA binaries install (check CUDA version and edit accordingly)
```
# 2. Add CUDA 12.8 repo for RHEL/AlmaLinux 8
sudo dnf install -y dnf-plugins-core
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf clean all

# 3. Install CUDA toolkit 12.8
sudo dnf install -y cuda-toolkit-12-8

# 4. Add to PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version  # should show 12.8
gcc --version   # should show 8.x or higher
```

#### Side note. Set this to emulate cuda environment before pip install
```
export CUDA_VISIBLE_DEVICES=""
export TORCH_CUDA_ARCH_LIST="7.0"
```

### Install podman
```
sudo dnf install -y podman
podman --version
```

### Then install uv and create python environment
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.13
uv venv --python 3.13 $HOME/rhel-env
source $HOME/rhel-env/bin/activate
```

### Then install python libraries
```
uv pip install ninja packaging wheel
uv pip install unsloth
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install "torchvision>=0.26.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
uv pip install \
    numpy pandas scikit-learn matplotlib seaborn \
    imbalanced-learn statsmodels \
    sentence-transformers umap-learn \
    xgboost nltk shap \
    jupyter notebook ipykernel ipywidgets \
    transformers accelerate datasets \
    huggingface-hub plotly fastapi \
    uvicorn pydantic python-dotenv tqdm openai \
    psycopg2-binary sqlalchemy asyncpg \
    alembic pgvector \
    trl peft bitsandbytes \
    tensorboard scipy \
    sentencepiece protobuf \
    safetensors tokenizers \
    rouge-score evaluate
```

### Pack python env
```
tar -cf $HOME/rhel-env.tar $HOME/rhel-env
```

### Install postgres container
```
podman pull docker.io/library/postgres:18
podman save docker.io/library/postgres:18 -o $HOME/postgres.tar
```

### Install vLLM container
```
podman pull docker.io/vllm/vllm-openai:latest
podman save docker.io/vllm/vllm-openai:latest \
    -o $HOME/vllm.tar
```

### Create llama.cpp build with GPU support for RHEL compatible kernel 
> Haven't verified if it works. Might get stuck
#### First download llama.cpp
```
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```
#### Then build it (params might change for different GPU kernels)
```
# 4. Build with CUDA for V100 (sm70)
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=$(which nvcc)
cmake --build build --config Release -j$(nproc)
```
#### And pack it
```
# 5. Verify binaries exist
ls build/bin/
# should show: llama-cli, llama-server, llama-gguf etc.

# 6. Pack just the binaries
tar -cf $HOME/llama-cuda-sm70.tar \
    -C $HOME/llama.cpp/build/bin .
```

### Download hf models for vLLM (add HF_TOKEN env variable for higher download limits)
```
# First activate virtual env from above
source $HOME/rhel-env/bin/activate

export HF_TOKEN=<insert-HF-token-here>

# Gemma 4
hf download cyankiwi/gemma-4-31B-it-AWQ-8bit \
    --local-dir $HOME/models/gemma-4-31B-AWQ-8bit

# Package it
tar -cf $HOME/gemma-4-31B-AWQ-8bit.tar \
    -C $HOME/models gemma-4-31B-AWQ-8bit

# Qwen 3.6
hf download QuantTrio/Qwen3.6-27B-AWQ \
    --local-dir $HOME/models/Qwen3.6-27B-AWQ

tar -cf $HOME/Qwen3.6-27B-AWQ-4bit.tar \
    -C $HOME/models Qwen3.6-27B-AWQ
```
> See below for why we chose AWQ
#### NVIDIA V100 Precision Compatibility

| Precision | Type | V100 Compatible | Notes |
|---|---|---|---|
| FP32 | Floating Point 32-bit | ✅ Yes | Full support, slowest |
| FP16 | Floating Point 16-bit | ✅ Yes | Native tensor cores, recommended |
| INT8 | Integer 8-bit | ✅ Yes | Via GPTQ/AWQ, use `--dtype float16` |
| INT4 | Integer 4-bit | ✅ Yes | Via GPTQ/AWQ, use `--dtype float16` |
| BF16 | Brain Float 16-bit | ❌ No | Ampere (A100) minimum |
| FP8 | Floating Point 8-bit | ❌ No | Hopper (H100) minimum |
| FP4 / NVFP4 | Floating Point 4-bit | ❌ No | Blackwell (B100/RTX5090) minimum |
| INT3 | Integer 3-bit | ⚠️ Limited | GGUF only via llama.cpp, not vLLM |
| INT2 | Integer 2-bit | ⚠️ Limited | GGUF only, significant quality loss |

#### Key Rules for V100

| Rule | Detail |
|---|---|
| Always set | `--dtype float16` in vLLM |
| Never use | BF16 models — will error or fall back to FP32 |
| Check AWQ models | Some have BF16 activations despite INT8 weights — avoid these |
| Safe formats | AWQ INT4, AWQ INT8, GPTQ INT4, GPTQ INT8 — all with `--dtype float16` |
| Max VRAM | 64GB across 2x V100 32GB |

#### Quantization Quality vs Size Reference

| Format | Quality vs FP16 | Approx Size (27-32B model) | V100 |
|---|---|---|---|
| FP16 | 100% baseline | ~54-64GB | ✅ (barely fits) |
| BF16 | ~100% | ~54-64GB | ❌ |
| FP8 | ~99.5% | ~27-32GB | ❌ |
| GPTQ INT8 | ~99% | ~27-35GB | ✅ |
| AWQ INT8 | ~99% | ~27-35GB | ✅ |
| GPTQ INT4 | ~96% | ~14-18GB | ✅ |
| AWQ INT4 | ~95% | ~14-18GB | ✅ |
| NVFP4 | ~97% | ~14-16GB | ❌ |
| GGUF Q8_0 | ~99% | ~27-35GB | ✅ (llama.cpp only) |
| GGUF Q4_K_M | ~95% | ~14-18GB | ✅ (llama.cpp only) |
| GGUF Q2_K | ~85% | ~8-10GB | ✅ (llama.cpp only) |

#### GPU Architecture Reference

| GPU | Architecture | Compute | FP16 | BF16 | FP8 | FP4 |
|---|---|---|---|---|---|---|
| V100 | Volta | sm70 | ✅ | ❌ | ❌ | ❌ |
| A100 | Ampere | sm80 | ✅ | ✅ | ❌ | ❌ |
| H100 | Hopper | sm90 | ✅ | ✅ | ✅ | ❌ |
| RTX 4090 | Ada | sm89 | ✅ | ✅ | ✅ | ❌ |
| B100/RTX 5090 | Blackwell | sm120 | ✅ | ✅ | ✅ | ✅ |
