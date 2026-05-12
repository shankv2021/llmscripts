## Steps to install python, vllm, postgres environment in Redhat Enterprise linux env

> Before any of this check CUDA version, GCC version of your target linux server

### OS level installs (GCC and build tools)
`
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y gcc gcc-c++ make git wget curl which
`

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
tar -czf $HOME/rhel-env.tar.gz $HOME/rhel-env
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
