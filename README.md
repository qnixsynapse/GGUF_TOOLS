# GGUF Tools

This repository contains a collection of small, **pure‑Python** utilities that help you
inspect, edit, and compare GGUF (Graph Neural Network Universal Format) files.  The
tools are geared toward researchers and engineers working with LLM quantization and
conversion pipelines.  All scripts are intentionally lightweight – they only depend
on the standard library, *numpy* and *gguf-py* (plus optional *torch* for GPU
acceleration).

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/qnixsynapse/GGUF_TOOLS.git
cd GGUF_TOOLS

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install the Python dependencies
pip install -r requirements.txt
```

> **Tip**: Most scripts are self‑contained and only require the standard library. If you encounter missing dependencies, add them to a `requirements.txt` file.

## 📂 Directory Overview

```
├── ggml-inspect_v2.py                # Inspect GGUF headers, tensors, and metadata
├── gguf_editor_standalone.py         # Lightweight Tkinter GUI for editing tensors/metadata
├── gguf_manager.py                   # CLI helper – dump metadata, set values, and export
├── gguf_manager_tui_inplace.py       # Terminal UI for editing metadata on‑the‑fly
├── weight_compare.py                # Compute L2 / cosine differences between two GGUF checkpoints (CPU)
├── weight_compare_gpu.py            # GPU‑accelerated weight comparison (requires PyTorch)
├── Per-tensor quant mix vs Unsloth IQ4_XS (Qwen3.5-9B).md
└── README.md
```

### Script Highlights

* **`ggml-inspect_v2.py`** – Inspect a GGUF file and display the header, tensor shapes, data types and optional metadata.
* **`gguf_editor_standalone.py`** – Tkinter‑based GUI for viewing and editing tensors/metadata in place.  No external dependencies are required.
* **`gguf_manager.py`** – Simple command‑line tool: dump metadata (`--dump`), change scalar or string fields (`--set key=value`), or write a modified copy (`--output`).
* **`gguf_manager_tui_inplace.py`** – A lightweight terminal UI that lets you edit metadata on‑the‑fly without any external libraries.
* **`weight_compare.py` / `weight_compare_gpu.py`** – Compute L2 / cosine differences between two GGUF checkpoints.  The GPU variant uses *PyTorch* for fast dequantisation and reductions.

## 🚀 Usage Examples

```bash
# Inspect a GGUF file
python ggml-inspect_v2.py path/to/model.gguf

# Dump metadata to the console
python gguf_manager.py path/to/model.gguf --dump

# Update a scalar field and write a new file
python gguf_manager.py path/to/model.gguf --set "general.architecture=MyModel" --output model_modified.gguf

# Compare two checkpoints on the CPU
python weight_compare.py model_ref.gguf model_test.gguf

# Compare two checkpoints on a GPU (requires PyTorch)
python weight_compare_gpu.py model_ref.gguf model_test.gguf
```

### GUI Editing

```bash
python gguf_editor_standalone.py path/to/model.gguf
```

A simple window will pop up showing tensors and metadata.  You can modify values and save back to the same file.

## 🤝 Contributing

Feel free to open issues or pull requests. When submitting code, please:
1. Add a short description of the feature/bugfix.
2. Update the README if new functionality is added.
3. Include unit tests if applicable.

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions, reach out to the repository owner or open an issue.

---

**Happy hacking!**

