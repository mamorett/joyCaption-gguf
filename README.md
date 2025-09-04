# JoyCaption GGUF - Image Captioning GUI

JoyCaption is a desktop GUI for image captioning using GGUF models with llama-cpp-python. It provides a user-friendly interface for analyzing images and generating detailed captions using multimodal LLMs.

## Features

- Tkinter-based GUI for easy image selection and analysis
- Drag & drop support (optional, via `tkinterdnd2`)
- Supports multiple image formats: JPG, PNG, BMP, WEBP
- Customizable prompts via `.env` file
- Progress bar and batch processing
- Copy and save results

## Requirements

- Python 3.8+
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (GGUF model support)
- [Pillow](https://pypi.org/project/Pillow/)
- [pyperclip](https://pypi.org/project/pyperclip/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [tkinterdnd2](https://pypi.org/project/tkinterdnd2/) (optional, for drag & drop)

## Installation

You can install all required libraries using the provided script:

```bash
sh install_libs.sh
```

Or manually:

```bash
pip install pillow pyperclip python-dotenv
# Optional for drag & drop:
pip install tkinterdnd2
# For GGUF model (CPU only):
pip install llama-cpp-python
# For GPU (CUDA):
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

## Usage

1. Place your GGUF model and mmproj files in a known location.
2. Run the GUI:
	```bash
	python joycaption_ui.py
	```
3. In the app:
	- Click "Select Model" to choose your GGUF model file.
	- Click "Select mmproj" to choose your mmproj file.
	- Load the model.
	- Select image files or folders (drag & drop if supported).
	- Choose or enter a prompt (custom or from `.env`).
	- Click "Analyze Images" to generate captions.
	- Copy or save results as needed.

## Custom Prompts

You can define custom prompts in a `.env` file. Any environment variable ending with `_PROMPT` will be available in the prompt dropdown.

Example `.env`:

```
DETAILED_PROMPT=Describe the image in detail, including objects, actions, and context.
SHORT_PROMPT=Give a brief caption for the image.
```

## Notes

- Drag & drop requires `tkinterdnd2`. If not installed, use the file/folder browse buttons.
- GPU acceleration is available if installed with CUDA support.
- The app will warn about missing dependencies and provide install hints.

## License

See LICENSE for details.