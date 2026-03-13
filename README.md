# Smart Translator – Advanced Deep Learning Team 5

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Manager](https://img.shields.io/badge/Dependency%20Manager-uv-purple?style=flat-square)
![AI](https://img.shields.io/badge/AI-Ollama%20%7C%20PyTorch-orange?style=flat-square)

**Smart Translator** is an Agentic AI system designed to translate document images while preserving their original layout, fonts, and visual structure. Orchestrated by a local LLM supervisor via the Model Context Protocol (MCP), the system coordinates specialized tools for OCR, translation, font synthesis, and image inpainting. Additionally, it features a RAG-based "Chat-with-Document" capability.

---

## Project Structure

The project is modularized into specific agents and tools:

```text
├── assets/                  # Static assets and example images
├── checkpoints/ 
│    └── resnet_50/ 
│         └── best.ckpt      # Pretrained weights for document classification  
├── data/                    # Data storage for local processing
├── reports/                 # Project documentation
├── src/                     # Source code modules
│   ├── document_class_detector/  # PyTorch model for document classification (RVL-CDIP)
│   ├── document_image_renderer/  # Re-renders text into the image (Inpainting + Drawing)
│   ├── document_translator/      # LLM-based text translation tool
│   ├── font_detector/            # Font identification and size regression (MLP)
│   ├── image_provider/           # Handles file loading and webcam input
│   ├── layout_detector/          # Extract bounding boxes (OCR/Paddle/DocTR)
│   ├── rag_component_x/          # RAG "Chat with Document" feature
│   ├── supervisor/               # The Agentic Brain (LLM) controlling the flow
│   └── translator/               # Contains the main UI application
├── pyproject.toml           # Dependency definitions (uv)
├── uv.lock                  # Locked dependencies
└── README.md                # This file
```

---

## Archi
![alt text](<archi.png>)

---

## Prerequisites

Before running the project, ensure you have the following installed:
1. **Python Environment Manager (uv):** We use uv for fast dependency management.
2. **Git:** To clone the repository.
3. **Ollama:** Required to run the local LLMs (Supervisor & Translator). Download Ollama.
4. **Hardware:** A GPU (NVIDIA CUDA) is highly recommended. Running OCR, Inpainting, and LLMs purely on CPU will be significantly slower.

---

## Installation & Setup

1. **Install uv (Environment Manager)** 

If you haven't installed uv yet, run:
```
pip install uv
```

2. **Clone the Repository**

```
# Via HTTPS
git clone https://gitlab.lrz.de/advanceddl/advanceddlteam5.git

# OR Via SSH
git clone git@gitlab.lrz.de:advanceddl/advanceddlteam5.git

cd advanceddlteam5
```

3. **Sync Environment** 

Create the virtual environment and install all Python dependencies (PyTorch, DocTR, LangChain, Gradio, etc.) automatically:

```
uv sync
```

*This reads uv.lock and sets up the .venv directory.*

---

4. **Load LFS files**

Git LFS (Document Classifier)

This project uses Git-LFS to store the classification models.
Installation:

```
git lfs install
```

Download LFS files:

```
git lfs pull
```

The README of the Document Classifier is stored separately, otherwise this README would be overloaded.

Document Classifier README:
`src/document_class_detector/README.md`


## Model Setup (Critical Step)

This system relies on both local LLMs (via Ollama) and Deep Learning models (Python libraries).

1. **Ensure Ollama is Installed:**

This application requires a local Ollama instance to run the language model qwen3. Here are the steps to set it up:

**Install Ollama** - Visit the official website: **https://ollama.com/**

Download the version for your operating system (macOS, Windows, or Linux) and install it.

2. **Download/Pull Ollama Models** 

The Supervisor and Translation agents communicate with local LLMs. You must pull these models specifically:

Open your terminal (or Command Prompt/PowerShell on Windows).
type the following command to download the qwen3 model. This only needs to be done once:

```
# Start Ollama (if not running) and pull the model:
ollama pull qwen3      
```

3. **Automatic Python Model Downloads** 

The following models will be downloaded automatically on the first run of the Python script.

**Note:** The first execution will take several minutes to download these artifacts (approx. 2-3 GB).

- **SimpleLama:** Used for removing original text from images (Inpainting).
- **DocTR / OCR Predictor:** Used for detecting text layout and content.
- **HuggingFace Embeddings**: Used for the RAG component.


---

## Usage
The project includes a graphical user interface (GUI) based on Gradio.

**Run the Chat Application**
Execute the main UI script using uv:

```
uv run src/translator/ui_chat_app.py
```

**What happens next?**
1. **Server Start:** The terminal will show a FastMCP banner.
2. **Wait:** Wait until you see the log message:
    **INFO:** Uvicorn running on http://localhost:8000
3. **Open UI:** Open your web browser and navigate to: http://localhost:7860

*(The backend tool server runs on port 8000, but the user interface is on port 7860).*

---

## Dependency Management (For Developers)
This project uses uv to manage pyproject.toml.

- **Add a new package:**

```
uv add package_name
```

- **Update environment (e.g., after pulling changes from Git):**

```
uv sync
```

## Authors
Advanced Deep Learning - Team 5 \
*Hochschule München - University of Applied Sciences*

Christian Städter \
Christian Fabian \
Stefan Lutsch \
Chau Nguyen \
Manuel Pasti 