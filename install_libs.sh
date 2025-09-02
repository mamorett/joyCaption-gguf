# Basic requirements
pip install pillow pyperclip python-dotenv

# For drag & drop support (optional)
pip install tkinterdnd2

# For the GGUF model (choose one):
# CPU only:
#pip install llama-cpp-python

# GPU support (CUDA):
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# GPU support (Metal on macOS):
#CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

