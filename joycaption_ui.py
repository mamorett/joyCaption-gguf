import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import pyperclip
from typing import Dict, Any, List, Optional
import threading
import glob
import warnings
import io
from pathlib import Path
from dotenv import load_dotenv
from llama_cpp.llama_chat_format import Llava15ChatHandler


# Suppress all warnings
warnings.filterwarnings('ignore')

# Try to import tkinterdnd2 for drag and drop
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False


# JoyCaption Configuration
CONFIG = {
    'model_path': '/bidone/llama-joycaption-beta-one-hf-llava.Q6_K.gguf',
    'mmproj_path': '/bidone/llama-joycaption-beta-one-llava-mmproj-model-f16.gguf',
    'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
    'max_retries': 3,
    'max_image_size': 4096 * 4096,
    'max_file_size': 10 * 1024 * 1024,
    'max_new_tokens': 256,
    'n_ctx': 4096,
    'n_gpu_layers': -1,
    'verbose': False
}


class JoyCaptionImageAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JoyCaption Image Analyzer")
        self.root.geometry("1000x950")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Initialize model variables
        self.model = None
        self.model_loaded = False
        
        # Get prompts from .env
        self.prompts = self.get_prompts_from_env()
        
        self.setup_ui()
        
    def get_prompts_from_env(self):
        """Get all prompts from .env file that end with _PROMPT."""
        try:
            load_dotenv()
            prompts = {}
            for key, value in os.environ.items():
                if key.endswith('_PROMPT'):
                    prompt_name = key.replace('_PROMPT', '').lower().replace('_', '-')
                    prompts[prompt_name] = value
            return prompts
        except:
            return {}
        
    def select_mmproj_path(self):
        file_path = filedialog.askopenfilename(
            title="Select LLaVA mmproj GGUF",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if file_path:
            CONFIG['mmproj_path'] = file_path
            self.mmproj_path_var.set(file_path)        
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="JoyCaption Image Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Model status and load button
        model_frame = ttk.LabelFrame(main_frame, text="Model Status", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        self.model_status_var = tk.StringVar()
        self.model_status_var.set("Model not loaded")
        
        ttk.Label(model_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_status_label = ttk.Label(model_frame, textvariable=self.model_status_var, 
                                           foreground='red', font=('Arial', 9))
        self.model_status_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        # Model path selection
        self.model_path_btn = ttk.Button(model_frame, text="Select Model", command=self.select_model_path)
        self.model_path_btn.grid(row=0, column=2, padx=(0, 5))
        
        self.load_model_btn = ttk.Button(model_frame, text="Load Model", command=self.load_model_thread)
        self.load_model_btn.grid(row=0, column=3)
        
        # Model path display
        self.model_path_var = tk.StringVar()
        self.model_path_var.set(CONFIG['model_path'])
        ttk.Label(model_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.model_path_label = ttk.Label(model_frame, textvariable=self.model_path_var, 
                                         foreground='gray', font=('Arial', 8))
        self.model_path_label.grid(row=1, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))


        # In setup_ui() under Model path selection
        self.mmproj_path_var = tk.StringVar()
        self.mmproj_path_var.set(CONFIG.get('mmproj_path', 'Not set'))
        ttk.Label(model_frame, text="mmproj:").grid(row=2, column=0, sticky=tk.W, padx=(0,5), pady=(5,0))
        self.mmproj_path_label = ttk.Label(model_frame, textvariable=self.mmproj_path_var, foreground='gray', font=('Arial', 8))
        self.mmproj_path_label.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(5,0))
        self.mmproj_path_btn = ttk.Button(model_frame, text="Select mmproj", command=self.select_mmproj_path)
        self.mmproj_path_btn.grid(row=2, column=3)

        
        # Prompt configuration section
        prompt_frame = ttk.LabelFrame(main_frame, text="Prompt Configuration", padding="10")
        prompt_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        prompt_frame.columnconfigure(1, weight=1)
        
        # Preset prompts dropdown
        ttk.Label(prompt_frame, text="Preset:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.preset_var = tk.StringVar()
        preset_values = ["Custom"] + list(self.prompts.keys())
        self.preset_combo = ttk.Combobox(prompt_frame, textvariable=self.preset_var, 
                                        values=preset_values, state="readonly", width=20)
        self.preset_combo.set("Custom")
        self.preset_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        self.preset_combo.bind('<<ComboboxSelected>>', self.on_preset_change)
        
        # Custom prompt entry
        ttk.Label(prompt_frame, text="Prompt:").grid(row=1, column=0, sticky=(tk.W, tk.N), padx=(0, 5), pady=(5, 0))
        
        self.prompt_var = tk.StringVar()
        self.prompt_var.set("Describe this image in detail")
        
        self.prompt_text = tk.Text(prompt_frame, height=3, wrap=tk.WORD, font=('Arial', 10))
        self.prompt_text.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(5, 0))
        self.prompt_text.insert(1.0, "Describe this image in detail")
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Select Image File(s)", padding="10")
        file_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # File path display
        self.file_path_var = tk.StringVar()
        self.file_path_var.set("No file selected")
        
        ttk.Label(file_frame, text="File(s):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.file_label = ttk.Label(file_frame, textvariable=self.file_path_var, 
                                   foreground='gray', font=('Arial', 9))
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Browse buttons frame
        browse_frame = ttk.Frame(file_frame)
        browse_frame.grid(row=0, column=2, padx=(5, 0))
        
        self.browse_file_btn = ttk.Button(browse_frame, text="Browse File", command=self.browse_file)
        self.browse_file_btn.grid(row=0, column=0, padx=(0, 2))
        
        self.browse_folder_btn = ttk.Button(browse_frame, text="Browse Folder", command=self.browse_folder)
        self.browse_folder_btn.grid(row=0, column=1)
        
        # Drop zone
        drop_text = "Drag & Drop image file(s) or folder here" if HAS_DND else "Click here to select image file(s)"
        if not HAS_DND:
            drop_text += "\n(Install tkinterdnd2 for drag & drop: pip install tkinterdnd2)"
        
        self.drop_frame = tk.Frame(file_frame, bg='#e8e8e8', relief='ridge', bd=2, height=100)
        self.drop_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        self.drop_frame.grid_propagate(False)
        
        self.drop_label = tk.Label(self.drop_frame, text=drop_text, 
                                  bg='#e8e8e8', fg='gray', font=('Arial', 10),
                                  justify='center')
        self.drop_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Setup drag and drop if available
        if HAS_DND:
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind('<<Drop>>', self.on_drop)
            self.drop_frame.dnd_bind('<<DragEnter>>', self.on_drag_enter)
            self.drop_frame.dnd_bind('<<DragLeave>>', self.on_drag_leave)
        else:
            # Fallback to click
            self.drop_frame.bind('<Button-1>', self.browse_file)
            self.drop_label.bind('<Button-1>', self.browse_file)
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Text area for results
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, 
                                                     height=15, font=('Arial', 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Buttons frame
        buttons_frame = ttk.Frame(results_frame)
        buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        buttons_frame.columnconfigure(0, weight=1)
        
        # Status and buttons
        self.status_var = tk.StringVar()
        status_text = "Ready - Load model and select image file(s)" if HAS_DND else "Ready - Load model and select image file(s)"
        self.status_var.set(status_text)
        
        self.status_label = ttk.Label(buttons_frame, textvariable=self.status_var, 
                                     foreground='gray', font=('Arial', 9))
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Button frame
        btn_frame = ttk.Frame(buttons_frame)
        btn_frame.grid(row=1, column=0, sticky=tk.E)
        
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze Images", 
                                     command=self.analyze_images, state='disabled')
        self.analyze_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.copy_btn = ttk.Button(btn_frame, text="Copy Results", 
                                  command=self.copy_to_clipboard, state='disabled')
        self.copy_btn.grid(row=0, column=1, padx=(0, 5))
        
        self.save_btn = ttk.Button(btn_frame, text="Save to File", 
                                  command=self.save_to_file, state='disabled')
        self.save_btn.grid(row=0, column=2, padx=(0, 5))
        
        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_results)
        self.clear_btn.grid(row=0, column=3)
        
        # Progress bar (hidden by default)
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.progress.grid_remove()  # Hide initially
        
        # Store current files
        self.current_files = []
        self.current_results = []
    
    def select_model_path(self):
        """Select GGUF model file"""
        file_path = filedialog.askopenfilename(
            title="Select GGUF Model File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        
        if file_path:
            CONFIG['model_path'] = file_path
            self.model_path_var.set(file_path)
    
    def on_preset_change(self, event=None):
        """Handle preset prompt selection"""
        selected = self.preset_var.get()
        if selected != "Custom" and selected in self.prompts:
            self.prompt_text.delete(1.0, tk.END)
            self.prompt_text.insert(1.0, self.prompts[selected])
    
    def validate_image(self, image_path):
        """Validate image file"""
        try:
            with Image.open(image_path) as img:
                if img.size[0] * img.size[1] > CONFIG['max_image_size']:
                    raise ValueError("Image dimensions too large")
                if os.path.getsize(image_path) > CONFIG['max_file_size']:
                    raise ValueError("File size too large")
            return True
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")
    
    
    def load_model_thread(self):
        """Load model in separate thread"""
        if self.model_loaded:
            return
            
        self.model_status_var.set("Loading model...")
        self.model_status_label.configure(foreground='orange')
        self.load_model_btn.configure(state='disabled')
        
        def load_model():
            try:
                self.model = self.load_model_and_tokenizer()
                self.root.after(0, self.on_model_loaded)
            except Exception as e:
                self.root.after(0, self.on_model_error, str(e))
        
        thread = threading.Thread(target=load_model)
        thread.daemon = True
        thread.start()
    
    def on_model_loaded(self):
        """Handle successful model loading"""
        self.model_loaded = True
        self.model_status_var.set("Model loaded successfully")
        self.model_status_label.configure(foreground='green')
        self.load_model_btn.configure(text="Model Loaded", state='disabled')
        self.analyze_btn.configure(state='normal')
        self.status_var.set("Ready - Select image file(s) to analyze")
    
    def on_model_error(self, error_message):
        """Handle model loading error"""
        self.model_status_var.set(f"Error loading model: {error_message}")
        self.model_status_label.configure(foreground='red')
        self.load_model_btn.configure(state='normal')
        messagebox.showerror("Model Error", f"Failed to load model:\n{error_message}")
    
    def on_drop(self, event):
        """Handle file drop event"""
        files = self.root.tk.splitlist(event.data)
        if files:
            self.load_files(files)
        
        # Reset drop zone appearance
        self.drop_frame.configure(bg='#e8e8e8')
        self.drop_label.configure(bg='#e8e8e8', fg='gray')
    
    def on_drag_enter(self, event):
        """Handle drag enter event"""
        self.drop_frame.configure(bg='#d0f0d0')  # Light green
        self.drop_label.configure(bg='#d0f0d0', fg='#006600', text="Drop image file(s) or folder here!")
    
    def on_drag_leave(self, event):
        """Handle drag leave event"""
        self.drop_frame.configure(bg='#e8e8e8')
        self.drop_label.configure(bg='#e8e8e8', fg='gray', text="Drag & Drop image file(s) or folder here")
        
    def browse_file(self, event=None):
        """Open file dialog to select image file(s)"""
        file_paths = filedialog.askopenfilenames(
            title="Select Image File(s)",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")]
        )
        
        if file_paths:
            self.load_files(list(file_paths))
    
    def browse_folder(self):
        """Open folder dialog to select directory with image files"""
        folder_path = filedialog.askdirectory(title="Select Folder with Image Files")
        
        if folder_path:
            # Find all image files in the folder
            image_files = []
            for ext in CONFIG['supported_formats']:
                image_files.extend(glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True))
                image_files.extend(glob.glob(os.path.join(folder_path, f"**/*{ext.upper()}"), recursive=True))
            
            if image_files:
                self.load_files(image_files)
            else:
                messagebox.showinfo("No Files", "No image files found in the selected folder.")
    
    def load_files(self, file_paths):
        """Load and validate the selected files"""
        # Filter for image files and existing files
        valid_files = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # If it's a directory, find image files in it
                for ext in CONFIG['supported_formats']:
                    valid_files.extend(glob.glob(os.path.join(file_path, f"**/*{ext}"), recursive=True))
                    valid_files.extend(glob.glob(os.path.join(file_path, f"**/*{ext.upper()}"), recursive=True))
            elif os.path.exists(file_path):
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in CONFIG['supported_formats']:
                    valid_files.append(file_path)
        
        if not valid_files:
            messagebox.showwarning("Warning", "No valid image files found")
            return
        
        # Store files
        self.current_files = valid_files
        
        # Update UI
        if len(valid_files) == 1:
            self.file_path_var.set(os.path.basename(valid_files[0]))
        else:
            self.file_path_var.set(f"{len(valid_files)} image files selected")
        
        self.status_var.set(f"Ready - {len(valid_files)} image(s) selected")
    
    def analyze_images(self):
        """Analyze the selected images"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load the model first")
            return
        
        if not self.current_files:
            messagebox.showwarning("Warning", "Please select image files first")
            return
        
        # Get current prompt
        current_prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not current_prompt:
            messagebox.showwarning("Warning", "Please enter a prompt")
            return
        
        self.status_var.set("Analyzing images...")
        self.progress.grid()
        self.progress.start()
        
        # Disable buttons during processing
        self.analyze_btn.configure(state='disabled')
        self.browse_file_btn.configure(state='disabled')
        self.browse_folder_btn.configure(state='disabled')
        
        # Process files in separate thread
        thread = threading.Thread(target=self.analyze_images_thread, args=(self.current_files, current_prompt))
        thread.daemon = True
        thread.start()
    
    def analyze_images_thread(self, file_paths, prompt):
        """Analyze images in separate thread"""
        try:
            results = []
            for file_path in file_paths:
                try:
                    result = self.process_single_image(file_path, prompt)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'file_path': file_path,
                        'success': False,
                        'error': str(e),
                        'response': None
                    })
            
            # Update UI in main thread
            self.root.after(0, self.update_results, results)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    

    def load_model_and_tokenizer(self):
        mmproj = CONFIG.get('mmproj_path')
        if not mmproj or not os.path.exists(mmproj):
            raise FileNotFoundError(f"mmproj file not found: {mmproj}")

        chat_handler = Llava15ChatHandler(clip_model_path=mmproj)

        llm = Llama(
            model_path=CONFIG['model_path'],
            chat_handler=chat_handler,
            n_ctx=CONFIG.get('n_ctx', 8192),      # consider 8192 or 16384
            n_gpu_layers=CONFIG.get('n_gpu_layers', -1),
            n_batch=CONFIG.get('n_batch', 1024),
            logits_all=False,
            verbose=CONFIG.get('verbose', False),
        )
        return llm


    def process_single_image(self, image_path, prompt):
        """Llava15ChatHandler: pass image and prompt as separate parts in the first user turn."""
        try:
            # Validate + prepare
            self.validate_image(image_path)
            proc_path = self.prepare_image_for_processing(image_path)

            import base64, mimetypes, os
            mime, _ = mimetypes.guess_type(proc_path)
            if mime is None:
                mime = "image/png"
            with open(proc_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            img_data_url = f"data:{mime};base64,{img_b64}"

            if proc_path != image_path and os.path.exists(proc_path):
                try:
                    os.remove(proc_path)
                except:
                    pass

            # Keep system concise but firm
            system_prompt = (
                "You are a grounded vision assistant. Only describe what is clearly visible. "
                "If anything is uncertain or not visible, say 'cannot determine'. Do not guess."
            )

            # IMPORTANT: structured parts; use key 'text' for the text part
            user_content = [
                {"type": "image_url", "image_url": {"url": img_data_url}},
                {"type": "text", "text": prompt.strip()},
            ]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            gen_kwargs = dict(
                messages=messages,
                max_tokens=CONFIG.get('max_new_tokens', 512),
                temperature=0.0,
                top_p=0.7,
                top_k=20,
                repeat_penalty=1.15,
                stop=["</s>", "Human:", "Assistant:", "ASSISTANT:", "\n\n"],
            )

            # Optional debug: inspect what’s being sent
            # import json; print("[DEBUG] messages:", json.dumps(messages, ensure_ascii=False)[:500], "...")

            resp = self.model.create_chat_completion(**gen_kwargs)
            answer = resp['choices'][0]['message']['content']

            # Clean role prefixes if they appear
            for tag in ("ASSISTANT:", "Assistant:", "Human:", "USER:", "User:"):
                if answer.startswith(tag):
                    answer = answer[len(tag):].lstrip()
            return {'file_path': image_path, 'success': True, 'error': None, 'response': answer.strip()}

        except Exception as e:
            return {'file_path': image_path, 'success': False, 'error': str(e), 'response': None}




    def prepare_image_for_processing(self, image_path):
        try:
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')

                max_dimension = 1024
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                ext = os.path.splitext(image_path)[1].lower()
                if ext == '.png':
                    temp_path = image_path + "_temp_resized.png"
                    img.save(temp_path, 'PNG', optimize=True)
                else:
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    temp_path = image_path + "_temp_resized.jpg"
                    img.save(temp_path, 'JPEG', quality=95, optimize=True)
                return temp_path
        except Exception as e:
            print(f"Warning: Image processing failed for {image_path}: {e}")
            return image_path

    
    def update_results(self, results):
        """Update UI with analysis results"""
        self.progress.stop()
        self.progress.grid_remove()
        self.analyze_btn.configure(state='normal')
        self.browse_file_btn.configure(state='normal')
        self.browse_folder_btn.configure(state='normal')

        # Defensive filtering
        safe_results = [r for r in results if isinstance(r, dict)]
        dropped = len(results) - len(safe_results)

        self.current_results = safe_results
        self.results_text.delete(1.0, tk.END)

        successful = 0
        all_responses = []

        for i, result in enumerate(safe_results):
            file_path = result.get('file_path', 'unknown')
            filename = os.path.basename(file_path) if file_path else 'unknown'

            if len(safe_results) > 1:
                self.results_text.insert(tk.END, f"=== {filename} ===\n")

            if result.get('success'):
                successful += 1
                response = result.get('response', '')
                self.results_text.insert(tk.END, f"{response}\n")
                all_responses.append(f"{filename}:\n{response}")
            else:
                error_msg = result.get('error', 'Unknown error')
                self.results_text.insert(tk.END, f"Error: {error_msg}\n")
                all_responses.append(f"{filename}:\nError: {error_msg}")

            if i < len(safe_results) - 1:
                self.results_text.insert(tk.END, "\n" + "="*60 + "\n\n")

        status = f"✓ Analyzed {successful}/{len(safe_results)} images successfully"
        if dropped > 0:
            status += f" (skipped {dropped} invalid result(s))"
        self.status_var.set(status)

        if successful > 0:
            self.copy_btn.configure(state='normal')
            self.save_btn.configure(state='normal')
            self.all_responses = all_responses
        else:
            self.all_responses = []

    
    def show_error(self, error_message):
        """Show error message"""
        self.progress.stop()
        self.progress.grid_remove()
        self.analyze_btn.configure(state='normal')
        self.browse_file_btn.configure(state='normal')
        self.browse_folder_btn.configure(state='normal')
        
        self.status_var.set(f"✗ Error: {error_message}")
        messagebox.showerror("Error", f"Failed to process images:\n{error_message}")
    
    def copy_to_clipboard(self):
        """Copy all results to clipboard"""
        if hasattr(self, 'all_responses') and self.all_responses:
            try:
                all_text = '\n\n'.join(self.all_responses)
                pyperclip.copy(all_text)
                self.status_var.set(f"✓ Results copied to clipboard!")
                
                # Reset status after 3 seconds
                self.root.after(3000, lambda: self.status_var.set("Ready"))
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy to clipboard:\n{e}")
    
    def save_to_file(self):
        """Save results to text file"""
        if hasattr(self, 'all_responses') and self.all_responses:
            # Default filename
            if len(self.current_files) == 1:
                
                base_name = os.path.splitext(os.path.basename(self.current_files[0]))[0]
                default_name = f"{base_name}_analysis.txt"
            else:
                default_name = f"image_analysis_{len(self.current_files)}_files.txt"
            
            file_path = filedialog.asksaveasfilename(
                title="Save Analysis Results",
                defaultextension=".txt",
                initialname=default_name,
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("JoyCaption Image Analysis Results\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"Generated on: {self.get_timestamp()}\n")
                        f.write(f"Model: {os.path.basename(CONFIG['model_path'])}\n")
                        f.write(f"Prompt: {self.prompt_text.get(1.0, tk.END).strip()}\n")
                        f.write(f"Files processed: {len(self.current_files)}\n\n")
                        f.write("=" * 50 + "\n\n")
                        
                        for response in self.all_responses:
                            f.write(response + "\n\n" + "-" * 30 + "\n\n")
                    
                    self.status_var.set(f"✓ Results saved to {os.path.basename(file_path)}")
                    
                    # Reset status after 3 seconds
                    self.root.after(3000, lambda: self.status_var.set("Ready"))
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save file:\n{e}")
    
    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def clear_results(self):
        """Clear all results and reset UI"""
        self.results_text.delete(1.0, tk.END)
        self.current_files = []
        self.current_results = []
        self.all_responses = []
        
        self.file_path_var.set("No file selected")
        self.status_var.set("Ready - Select image file(s) to analyze")
        
        # Disable result buttons
        self.copy_btn.configure(state='disabled')
        self.save_btn.configure(state='disabled')

def main():
    # Check if required packages are available
    missing_packages = []
    
    required_packages = ['pyperclip', 'Pillow']
    
    for package in required_packages:
        try:
            if package == 'Pillow':
                from PIL import Image
            elif package == 'pyperclip':
                import pyperclip
        except ImportError:
            missing_packages.append(package)
    
    # Check for llama-cpp-python
    if not HAS_LLAMA_CPP:
        missing_packages.append('llama-cpp-python')
    
    # Check for drag and drop support
    if not HAS_DND:
        print("Note: For drag & drop functionality, install tkinterdnd2:")
        print("pip install tkinterdnd2")
    
    if missing_packages:
        print("Missing required packages:", ', '.join(missing_packages))
        print("Please install them using:")
        for package in missing_packages:
            if package == 'llama-cpp-python':
                print("  # For CPU only:")
                print("  pip install llama-cpp-python")
                print("  # For GPU support (CUDA):")
                print("  CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python")
                print("  # For GPU support (Metal on macOS):")
                print("  CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python")
            else:
                print(f"  pip install {package}")
        return
    
    # Use TkinterDnD root if available, otherwise regular Tk
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    
    app = JoyCaptionImageAnalyzerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
