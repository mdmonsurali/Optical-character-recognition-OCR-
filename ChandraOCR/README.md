# Chandra OCR — PDF to DOCX Pipeline

Convert PDFs to structured Word documents using [Chandra OCR](https://github.com/datalab-to/chandra) running on a local vLLM server. Handles tables, handwriting, math, forms, and complex layouts.

## Repository Structure

```
├── chandra_vllm.py      # Main pipeline: PDF → HTML → DOCX
├── html_to_docx.py      # HTML to DOCX converter
├── local.env            # vLLM server config (not committed if sensitive)
├── documents/           # Input PDFs
├── output/              # Generated HTML and DOCX files
└── requirements.txt
```

## Requirements

- Python 3.10+
- NVIDIA GPU with **22GB+ VRAM** (A10G, A100, H100)
- CUDA 12.x

## Installation

**1. Clone and create virtual environment**

```bash
git clone https://github.com/mdmonsurali/Optical-character-recognition-OCR-/tree/main/ChandraOCR
cd chandra-ocr-pipeline

python3 -m venv chandra-ocr
source chandra-ocr/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
sudo apt-get install -y poppler-utils
```

**3. Start the vLLM server**

> First run downloads the model (~17GB). Parameters tuned for 22GB A10G — adjust `--gpu-memory-utilization` for larger cards.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
vllm serve datalab-to/chandra \
    --served-model-name chandra \
    --port 8009 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 4 \
    --enforce-eager \
    --dtype bfloat16
```

**4. Configure `local.env`**

```env
VLLM_API_BASE=http://localhost:8009/v1
VLLM_MODEL_NAME=chandra
MAX_OUTPUT_TOKENS=3000
```

**5. Run**

Place your PDF in `documents/`, update `PDF_PATH` in `chandra_vllm.py`, then:

```bash
python chandra_vllm.py
```

Output files (`<filename>.html` and `<filename>.docx`) are saved to `output/`.

## How It Works

```
PDF → page images (pdf2image) → Chandra OCR (vLLM) → HTML with layout blocks → DOCX (python-docx)
```

Chandra outputs HTML with `data-label` and `data-bbox` attributes per block. `html_to_docx.py` maps these to Word styles: tables → Table Grid, Section-Header → Heading 2, List-Group → List Bullet/Number, Equation-Block → Cambria Math, etc.
