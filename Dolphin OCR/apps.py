import streamlit as st
from PIL import Image
import io
import base64
from pdf2image import convert_from_bytes
from docx import Document
import json
import subprocess
import tempfile
import os

from dolphin_main import run_dolphin, json_to_docx  # Ensure this version supports `base_path`

# App setup
st.set_page_config(page_title="Dolphin Image Parser", page_icon="🐬", layout="wide")

# Logo and title
logo_path = "./image/dolphin.png"
logo_base64 = base64.b64encode(open(logo_path, "rb").read()).decode()
st.markdown(f"""
    # <img src="data:image/png;base64,{logo_base64}" width="80" style="vertical-align: -10px;"> Dolphin Image Parser
""", unsafe_allow_html=True)

st.markdown('<p style="margin-top: -20px;">Extract Markdown-structured text, JSON & DOCX from Images, Or PDFs files using the Dolphin model.</p>', unsafe_allow_html=True)
st.markdown("---")

# === SIDEBAR ===
with st.sidebar:
    st.header("📄 Upload File")

    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = None

    uploaded_file = st.file_uploader("Upload image, PDF or DOCX", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        
        # ✅ Only extract filename after upload
        original_filename = os.path.splitext(uploaded_file.name)[0]
        st.session_state["original_filename"] = original_filename

    if st.session_state["uploaded_file"]:
        if st.button("Extract Text 📄", type="primary"):
            uploaded = st.session_state["uploaded_file"]
            file_ext = uploaded.name.split('.')[-1].lower()

            st.session_state["ocr_result"] = None
            st.session_state["ocr_json"] = None
            st.session_state["file_preview"] = None

            if file_ext in ["png", "jpg", "jpeg"]:
                image = Image.open(uploaded)
                st.session_state["file_preview"] = image
                results, markdown = run_dolphin(image=image)
                st.session_state["ocr_result"] = markdown
                st.session_state["ocr_json"] = json.dumps(results, indent=2)

                docx_buffer = io.BytesIO()
                json_to_docx(results, output_path=docx_buffer, base_path=".")
                docx_buffer.seek(0)
                st.session_state["ocr_docx_bytes"] = docx_buffer

            elif file_ext == "pdf":
                try:
                    pdf_images = convert_from_bytes(uploaded.read())
                    st.session_state["file_preview"] = pdf_images
                    all_md, all_json = [], []
                    for idx, img in enumerate(pdf_images):
                        res, md = run_dolphin(image=img)
                        all_md.append(f"### Page {idx+1}\n{md}")
                        all_json.append(res)
                    st.session_state["ocr_result"] = "\n\n".join(all_md)
                    st.session_state["ocr_json"] = json.dumps(all_json, indent=2)

                    docx_buffer = io.BytesIO()
                    flattened_json = [block for page in all_json for block in page]
                    json_to_docx(flattened_json, output_path=docx_buffer, base_path=".")
                    docx_buffer.seek(0)
                    st.session_state["ocr_docx_bytes"] = docx_buffer

                except Exception as e:
                    st.error(f"PDF conversion error: {e}")


    # Preview after extract
    if "file_preview" in st.session_state and st.session_state["file_preview"]:
        st.subheader("📌 Preview")
        preview = st.session_state["file_preview"]
        if isinstance(preview, list):
            st.image(preview, caption=[f"Page {i+1}" for i in range(len(preview))], use_container_width=True)
        elif isinstance(preview, Image.Image):
            st.image(preview, caption="Uploaded Image", use_container_width=True)
        elif isinstance(preview, str):
            st.text_area("DOCX Content Preview", value=preview, height=300)

# === CLEAR BUTTON ===
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear 🚽"):
        for key in ["uploaded_file", "ocr_result", "ocr_json", "file_preview"]:
            st.session_state.pop(key, None)
        st.rerun()

# === MAIN DISPLAY ===
if st.session_state.get("ocr_result"):
    st.markdown("## 📄 Extracted Markdown")
    st.markdown(st.session_state["ocr_result"], unsafe_allow_html=True)
    
    original_name = st.session_state.get("original_filename", "dolphin_output")

    st.markdown("### 📅 Downloads")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="Download Markdown",
            data=st.session_state["ocr_result"],
            file_name=f"{original_name}.md",
            mime="text/markdown"
        )

    with col2:
        st.download_button(
            label="Download JSON",
            data=st.session_state["ocr_json"],
            file_name=f"{original_name}.json",
            mime="application/json"
        )

    with col3:
        if st.session_state.get("ocr_docx_bytes"):
        
            st.download_button(
                label="📝 Download DOCX",
                data=st.session_state["ocr_docx_bytes"],
                file_name=f"{original_name}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
else:
    st.info("Upload a document and click 'Extract Text' in the sidebar.")

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: gray;'>
    Made with ❤️ using <strong>Bytedance 🐬 Dolphin Image Parser</strong> |
    <a href="https://github.com/bytedance/Dolphin" target="_blank">GitHub</a> |
    © 2025<br>
    Author: <strong>Md Monsur Ali</strong> |
    <a href="https://inventaai.com/" target="_blank">Website</a> |
    <a href="https://www.linkedin.com/company/inventaai/" target="_blank">LinkedIn</a> |
    <a href="https://github.com/mdmonsurali" target="_blank">Author GitHub</a> |
    © 2025
</div>
""", unsafe_allow_html=True)

