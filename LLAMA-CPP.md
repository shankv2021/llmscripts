## Run llama-cpp on Windows

1. Download the latest [llama-cpp](https://github.com/ggml-org/llama.cpp/releases) zip file from GitHub.

2. Unzip the downloaded file, then run `cd llama-<version-name>-bin-win-cuda-12.4-x64/`

3. Have a local gguf model ready or download one from [huggingface](https://huggingface.co/google/gemma-3-4b-it)

4. Open cmd prompt and run `llama-server -m <path-to-model>` to start server
