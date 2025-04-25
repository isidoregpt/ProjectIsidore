# This program is licensed under the GNU General Public License v3.0.
# For more details, see: https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author: Jonathan Graziola (isidore.gpt@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import streamlit as st
import openai
from io import BytesIO
from pptx import Presentation
from openpyxl import load_workbook
import fitz  # PyMuPDF
from PIL import Image
import tempfile

# Streamlit app configuration
st.set_page_config(
    page_title="📑 Enhanced Private Equity Proofreader",
    layout="wide"
)

# --- Helper functions for file processing ---

def extract_pptx_content(file_bytes):
    prs = Presentation(BytesIO(file_bytes))
    text_blocks = []
    image_count = 0
    layout_issues = []
    for idx, slide in enumerate(prs.slides, start=1):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                slide_text.append(shape.text)
            if shape.shape_type == 13:  # Picture
                image_count += 1
        # Basic layout check: consistent title font size
        if prs.slide_width and prs.slide_height:
            # placeholder for detailed layout logic
            pass
        text_blocks.append(f"-- Slide {idx} Text --\n" + "\n".join(slide_text))
    return "\n\n".join(text_blocks), image_count, layout_issues


def extract_xlsx_content(file_bytes):
    wb = load_workbook(filename=BytesIO(file_bytes), data_only=True)
    sheet_texts = []
    style_issues = []
    for sheet in wb.worksheets:
        cells = []
        for row in sheet.iter_rows(min_row=1, max_row=20, max_col=10):
            row_text = [str(cell.value) for cell in row if cell.value is not None]
            if row_text:
                cells.append("\t".join(row_text))
        sheet_texts.append(f"-- Sheet: {sheet.title} --\n" + "\n".join(cells))
        # placeholder for style_issues analysis
    return "\n\n".join(sheet_texts), style_issues


def extract_pdf_content(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    image_count = 0
    for page in doc:
        pages_text.append(page.get_text())
        for img in page.get_images():
            image_count += 1
    return "\n\n".join(pages_text), image_count


# --- Main Application ---

def main():
    st.title("🕵️ Enhanced Private Equity Deal Proofreader")
    st.markdown(
        "Upload your Word, PPTX, XLSX, or PDF documents for a professional proofread focusing on text, visuals, and layout consistency."
    )

    # Sidebar: API Key & model selection
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
    temperature = st.sidebar.slider("Temperature", 0.0, 0.5, 0.2)

    # File upload
    uploaded = st.file_uploader(
        "Choose a document", type=["pptx", "xlsx", "pdf"], accept_multiple_files=False
    )

    if uploaded and api_key:
        content = uploaded.read()
        ext = uploaded.name.split('.')[-1].lower()

        # Extract content and metadata
        if ext == 'pptx':
            text, img_count, layout_issues = extract_pptx_content(content)
            metadata = f"Slides contain {img_count} images."
        elif ext == 'xlsx':
            text, style_issues = extract_xlsx_content(content)
            metadata = f"Workbook with {len(style_issues)} style issues detected."  # placeholder
        elif ext == 'pdf':
            text, img_count = extract_pdf_content(content)
            metadata = f"PDF contains {img_count} embedded images."
        else:
            st.error("Unsupported format.")
            return

        # Build system prompt with enhanced instructions
        system_prompt = (
            "You are an expert proofreader for private equity deal documents. "
            "Perform the following on the provided content:\n"
            "1. Correct grammar, spelling, and punctuation.\n"
            "2. Ensure consistent financial/legal terminology (e.g., EBITDA, enterprise value).\n"
            "3. Verify uniform formatting and layout consistency across slides/pages.\n"
            "4. Review visual elements: images, graphs, their descriptions, and placement.\n"
            "5. Flag ambiguous phrasing and suggest clarifications.\n"
            "6. Check date, version, and party-name consistency.\n"
            "7. Note any visual/layout integration issues (e.g., misaligned charts, inconsistent fonts)."
        )
        user_prompt = (
            f"Document Metadata: {metadata}\n\n"
            f"Extracted Text:\n{text}\n\n"
            "Please return:\n"
            "- Full corrected text \n"
            "- Summary of changes made to text and visuals. \n"
            "- List of any detected layout or style inconsistencies."
        )

        # Call OpenAI
        openai.api_key = api_key
        with st.spinner("Analyzing and proofreading..."):
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role":"system", "content":system_prompt},
                        {"role":"user", "content":user_prompt}
                    ],
                    temperature=temperature
                )
                result = resp.choices[0].message.content
            except Exception as e:
                st.error(f"Error: {e}")
                return

        # Display output
        st.subheader("📄 Proofread & Analysis Result")
        st.text_area("Result", result, height=400)

        # Download corrected text
        st.download_button(
            "Download Result as TXT",
            result,
            file_name="proofread_result.txt",
            mime="text/plain"
        )
    elif uploaded:
        st.warning("Enter your OpenAI API key to proceed.")

if __name__ == "__main__":
    main()
