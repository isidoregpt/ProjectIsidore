import streamlit as st
import datetime
import io
import zipfile
import requests
import os
from fpdf import FPDF
import pandas as pd
from openai import OpenAI

# Model provider configurations including parameter support
MODEL_OPTIONS = {
    "OpenAI": {
        "models": [
            "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14",
            "gpt-4.5-preview-2025-02-27", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18",
            "chatgpt-4o-latest", "o1-2024-12-17", "o3-mini-2025-01-31", "o4-mini-2025-04-16"
        ],
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_iterations": True,
        "temp_range": (0.0, 2.0)
    },
    "Cerebras": {
        "models": [
            "llama-4-scout-17b-16e-instruct", "llama3.1-8b", "llama-3.3-70b"
        ],
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_iterations": False,
        "temp_range": (0.0, 1.0)
    },
    "DeepSeek": {
        "models": ["deepseek-reasoner", "deepseek-chat"],
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_iterations": False,
        "temp_range": (0.0, 1.0)
    },
    "Anthropic": {
        "models": ["claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"],
        "supports_temperature": True,
        "supports_top_p": False,
        "supports_iterations": False,
        "temp_range": (0.0, 1.0)
    },
    "Gemini": {
        "models": ["gemini-2.0-flash-001", "gemini-2.0-flash-lite-001", "gemini-2.0-flash-lite"],
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_iterations": False,
        "temp_range": (0.0, 2.0)
    },
    "Grok": {
        "models": [
            "grok-2-latest", "grok-3-beta", "grok-3-fast-beta",
            "grok-3-mini-beta", "grok-3-mini-fast-beta"
        ],
        "supports_temperature": True,
        "supports_top_p": True,
        "supports_iterations": False,
        "temp_range": (0.0, 1.0)
    }
}

API_ENDPOINT = "http://localhost:8000/api/chat"
SAVED_RUBRICS = {
    "6 Traits": "rubrics/6_traits.txt",
    "SAT Rubric": "rubrics/sat_rubric.txt",
    "AP Lang Rubric": "rubrics/ap_lang.txt"
}

VECTOR_STORE_ID = None


def create_vector_store(api_key, prompt, rubric, reference):
    global VECTOR_STORE_ID
    if VECTOR_STORE_ID is None:
        try:
            client = OpenAI(api_key=api_key)
            vs = client.vector_stores.create(name="Essay Grading Context")
            VECTOR_STORE_ID = vs.id

            for name, content in {
                "prompt.txt": prompt,
                "rubric.txt": rubric,
                "reference.txt": reference
            }.items():
                with open(name, "w", encoding="utf-8") as f:
                    f.write(content)
                with open(name, "rb") as f:
                    client.vector_stores.files.upload_and_poll(vector_store_id=vs.id, file=f)
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False
    return True


def save_pdf(essay_name, model_outputs):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, f"Graded Essay: {essay_name}", ln=True, align="C")
    pdf.set_font("Arial", size=10)

    for model, text in model_outputs.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, f"Model: {model}", ln=True)
        pdf.set_font("Arial", size=10)
        for line in text.split("\n"):
            pdf.multi_cell(0, 5, line)
        pdf.ln()

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer


def display_comparison_table(results):
    total_scores = []
    rows = []
    for model, content in results.items():
        lines = content.splitlines()
        deduction_line = next((line for line in lines if "points deducted" in line.lower()), "Not Found")
        summary = next((line for line in lines[::-1] if line.strip()), "")
        try:
            points = float(''.join(filter(str.isdigit, deduction_line)))
        except:
            points = 0
        total_scores.append((model, points))
        rows.append({"Model": model, "Points Deducted": deduction_line, "Summary": summary[:150]})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    if total_scores:
        st.subheader("📊 Average Points Deducted")
        score_df = pd.DataFrame(total_scores, columns=["Model", "Points Deducted"])
        avg = score_df.groupby("Model")["Points Deducted"].mean().reset_index()
        st.table(avg)


def apply_model_parameters(selected_models, temperature, top_p, iterations):
    """Apply appropriate model parameters based on provider compatibility"""
    for model in selected_models:
        provider = model["provider"]
        model_config = MODEL_OPTIONS.get(provider, {})
        
        # Add temperature if supported
        if model_config.get("supports_temperature", False):
            min_temp, max_temp = model_config.get("temp_range", (0.0, 1.0))
            # Clamp temperature to model's valid range
            model["temperature"] = max(min_temp, min(temperature, max_temp))
        
        # Add top_p if supported
        if model_config.get("supports_top_p", False):
            model["top_p"] = top_p
        
        # Add iterations if supported
        if model_config.get("supports_iterations", False):
            model["iterations"] = iterations
    
    return selected_models


def main():
    st.set_page_config(page_title="Multi-Model Essay Grader", layout="wide")
    st.title("🤖 Multi-Model Essay Grader with Retrieval & Comparison")

    with st.expander("ℹ️ Instructions"):
        st.markdown("""
This tool grades essays using multiple AI models, compares outputs, and exports results.
1. Upload your Prompt, Rubric, Reference, and Essays
2. Select saved or custom rubric
3. Choose models and settings
4. Vector store context will improve consistency
5. Export results as PDF or ZIP
""")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        api_key_file = st.file_uploader("🔑 API Key (TXT)", type=["txt"])
    with col2:
        prompt_file = st.file_uploader("📝 Prompt (TXT)", type=["txt"])
    with col3:
        use_saved_rubric = st.selectbox("📋 Choose Saved Rubric", ["None"] + list(SAVED_RUBRICS.keys()))
    with col4:
        rubric_file = st.file_uploader("📤 Or Upload Custom Rubric (TXT)", type=["txt"])
    with col5:
        reference_file = st.file_uploader("📘 Reference Material (TXT)", type=["txt"])

    st.markdown("---")
    st.subheader("📂 Upload Essays")
    essay_zip = st.file_uploader("📦 Upload ZIP of Essays (optional)", type=["zip"])
    
    essay_files = []
    essay_names = []
    
    if essay_zip:
        with zipfile.ZipFile(essay_zip) as z:
            essay_names = [f for f in z.namelist() if f.endswith(".txt")]
            essay_files = [z.read(f).decode("utf-8") for f in essay_names]
    else:
        uploaded_files = st.file_uploader("📂 Upload one or more student essays", type=["txt"], accept_multiple_files=True)
        if uploaded_files:
            essay_names = [f.name for f in uploaded_files]
            essay_files = [f.read().decode("utf-8") for f in uploaded_files]

    st.subheader("🤖 Choose Models for Grading")
    selected_models = []
    for provider, provider_config in MODEL_OPTIONS.items():
        with st.expander(f"{provider} Models"):
            for model in provider_config["models"]:
                if st.checkbox(f"{model}", key=f"{provider}_{model}"):
                    selected_models.append({"provider": provider, "model": model})

    st.markdown("---")
    st.subheader("⚙️ Model Behavior Settings")
    st.info("Note: Settings will only be applied to models that support them.")
    
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, 
                           help="Controls randomness: Lower values are more deterministic, higher values more creative.")
    
    top_p = st.slider("Top-p Sampling", 0.1, 1.0, 0.9, 0.01, 
                     help="Controls diversity: Lower values consider only the most likely tokens.")
    
    iterations = st.number_input("Iterations", min_value=1, max_value=5, value=1, 
                               help="Number of grading iterations (only supported by select models).")

    if st.button("🚀 Grade Essays"):
        if not api_key_file or not prompt_file or not reference_file or not essay_files or not selected_models:
            st.error("Please upload all required files and select at least one model.")
        else:
            try:
                api_key = api_key_file.read().decode("utf-8").strip()
                prompt = prompt_file.read().decode("utf-8")
                reference = reference_file.read().decode("utf-8")

                if use_saved_rubric != "None":
                    with open(SAVED_RUBRICS[use_saved_rubric], "r", encoding="utf-8") as f:
                        rubric = f.read()
                elif rubric_file:
                    rubric = rubric_file.read().decode("utf-8")
                else:
                    st.error("Please select or upload a rubric.")
                    return

                # Create vector store context
                if not create_vector_store(api_key, prompt, rubric, reference):
                    return
                    
                all_pdfs = []

                for i, essay_text in enumerate(essay_files):
                    essay_name = essay_names[i] if i < len(essay_names) else f"essay_{i+1}.txt"
                    
                    # Apply appropriate model parameters based on provider compatibility
                    models_with_params = apply_model_parameters(
                        selected_models.copy(),  # Use copy to avoid modifying original selection
                        temperature, 
                        top_p, 
                        iterations
                    )
                    
                    request_payload = {
                        "message": f"Prompt:\n{prompt}\n\nReference:\n{reference}\n\nRubric:\n{rubric}\n\nEssay:\n{essay_text}",
                        "models": models_with_params,
                        "vector_store_id": VECTOR_STORE_ID
                    }
                    
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

                    with st.spinner(f"Grading {essay_name}..."):
                        response = requests.post(API_ENDPOINT, json=request_payload, headers=headers)
                        response.raise_for_status()
                        results = response.json().get("results", {})

                    st.subheader(f"📝 Results for {essay_name}")
                    display_comparison_table(results)
                    pdf = save_pdf(essay_name, results)
                    all_pdfs.append((essay_name, pdf))
                    st.download_button("📥 Download as PDF", data=pdf, file_name=f"graded_{essay_name}.pdf", mime="application/pdf")

                if len(all_pdfs) > 1:  # Only create ZIP if multiple essays
                    zip_output = io.BytesIO()
                    with zipfile.ZipFile(zip_output, "w", zipfile.ZIP_DEFLATED) as zipf:
                        for name, pdf in all_pdfs:
                            pdf.seek(0)  # Ensure we're at the start of the PDF data
                            zipf.writestr(f"{name}.pdf", pdf.read())
                    zip_output.seek(0)
                    st.download_button("📦 Download All as ZIP", data=zip_output, file_name="all_graded_essays.zip", mime="application/zip")

            except Exception as e:
                st.error(f"❌ Error grading essays: {e}")

if __name__ == "__main__":
    main()
