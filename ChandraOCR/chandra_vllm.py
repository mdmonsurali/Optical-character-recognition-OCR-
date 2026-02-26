# test_chandra_vllm_md_docx.py
import time
from pathlib import Path
from pdf2image import convert_from_path
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from html_to_docx import html_string_to_docx

PDF_PATH = "documents/EBK700_2000_SBU Komfort gedrehte Seiten.pdf"
OUTPUT_DIR = "./output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

manager = InferenceManager(method="vllm")
images = convert_from_path(PDF_PATH, dpi=150)
print(f"Pages: {len(images)}")

start = time.time()
all_pages = []

for i, image in enumerate(images):
    print(f"Processing page {i+1}/{len(images)}...")
    batch = [BatchInputItem(image=image, prompt_type="ocr_layout")]
    result = manager.generate(batch)[0]
    all_pages.append(result.markdown)

# Save combined HTML
input_stem = Path(PDF_PATH).stem
html_file = Path(OUTPUT_DIR) / f"{input_stem}.html"
html_file.write_text("\n\n".join(all_pages))

# Convert to DOCX
docx_path = Path(OUTPUT_DIR) / f"{input_stem}.docx"
html_string_to_docx("\n\n".join(all_pages), str(docx_path))

print(f"Done in {time.time() - start:.1f}s → {docx_path}")