import os, base64, mimetypes, sys
from PIL import Image
from llama_cpp import Llama

CONFIG = {
    "model_path": "/bidone/llama-joycaption-beta-one-hf-llava.Q6_K.gguf",
    "mmproj_path": "/bidone/llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
    "n_ctx": 4096,
    "n_gpu_layers": -1,   # set to -1 if you have GPU; else 0
    "n_batch": 1024,      # increase if you have VRAM/CPU
    "verbose": True,      # True for first test; then set False
}

def encode_image_to_data_url(path):
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def prepare_image(path, max_dim=1024):
    with Image.open(path) as img:
        mode = img.mode
        if mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        ext = os.path.splitext(path)[1].lower()
        tmp = path + (".temp.png" if ext == ".png" else ".temp.jpg")
        if ext == ".png":
            img.save(tmp, "PNG", optimize=True)
        else:
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(tmp, "JPEG", quality=95, optimize=True)
        return tmp

def load_model():
    kwargs = dict(
        model_path=CONFIG["model_path"],
        mmproj_path=CONFIG["mmproj_path"],
        n_ctx=CONFIG["n_ctx"],
        n_gpu_layers=CONFIG["n_gpu_layers"],
        n_batch=CONFIG["n_batch"],
        logits_all=False,
        verbose=CONFIG["verbose"],
    )
    # Try auto format first
    try:
        llm = Llama(**kwargs)
        print("Loaded model (auto chat format).")
        return llm
    except Exception as e1:
        print("Auto failed:", e1)
    # Fallback to chatml
    llm = Llama(chat_format="chatml", **kwargs)
    print("Loaded model (chatml).")
    return llm

def ask_image(llm, img_path, prompt="Describe what you see. Be literal. If unsure, say 'cannot determine'."):
    proc = prepare_image(img_path)
    data_url = encode_image_to_data_url(proc)
    try:
        os.remove(proc)
    except:
        pass

    messages = [
        {"role": "system", "content": "Only describe what is clearly visible. Do not guess."},
        {"role": "user", "content": f"{prompt}\n<image>", "images": [data_url]},
    ]

    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=160,
        temperature=0.0,
        top_p=0.65,
        top_k=20,
        min_p=0.1,
        repeat_penalty=1.15,
        stop=["</s>", "<|im_end|>"],
    )
    return out["choices"][0]["message"]["content"].strip()

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_joycaption_llamacpp.py /path/black_image.png /path/busy_image.jpg")
        sys.exit(1)

    imgA, imgB = sys.argv[1], sys.argv[2]
    assert os.path.exists(imgA), f"Missing: {imgA}"
    assert os.path.exists(imgB), f"Missing: {imgB}"

    llm = load_model()
    print("[DEBUG] metadata:", getattr(llm, "metadata", None))

    print("\n--- IMAGE A:", imgA)
    ansA = ask_image(llm, imgA)
    print(ansA)

    print("\n--- IMAGE B:", imgB)
    ansB = ask_image(llm, imgB)
    print(ansB)

    print("\nDIFFERENT? ", "YES" if ansA[:200] != ansB[:200] else "NO (likely vision not engaged)")

if __name__ == "__main__":
    main()
