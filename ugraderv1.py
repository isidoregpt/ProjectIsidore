import streamlit as st
import datetime
import io
import zipfile
import os
from fpdf import FPDF
import pandas as pd
from openai import OpenAI
import requests
import json
import time

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
        "temp_range": (0.0, 2.0)
    },
    "Anthropic": {
        "models": ["claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"],
        "supports_temperature": True,
        "supports_top_p": False,
        "temp_range": (0.0, 1.0)
    }
    # Other providers can be added here as needed
}

class ModelManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.openai_client = OpenAI(api_key=api_key)
        self.vector_store_id = None
        
    def create_vector_store(self, prompt, rubric, reference):
        """Create vector store for retrieval-augmented grading"""
        try:
            # Create a new vector store
            vs = self.openai_client.vector_stores.create(name="Essay Grading Context")
            self.vector_store_id = vs.id
            
            # Save context files temporarily
            context_files = {
                "prompt.txt": prompt,
                "rubric.txt": rubric,
                "reference.txt": reference
            }
            
            # Upload each file to the vector store
            for name, content in context_files.items():
                with open(name, "w", encoding="utf-8") as f:
                    f.write(content)
                with open(name, "rb") as f:
                    self.openai_client.vector_stores.files.upload_and_poll(
                        vector_store_id=self.vector_store_id, 
                        file=f
                    )
                    
                # Clean up temporary file
                try:
                    os.remove(name)
                except:
                    pass
                    
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False
    
    def grade_essay_with_openai(self, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        """Grade an essay using an OpenAI model with retrieval"""
        try:
            grading_request = (
                f"Prompt:\n{prompt}\n\n"
                f"Reference Material:\n{reference}\n\n"
                f"Rubric:\n{rubric}\n\n"
                f"Essay to grade:\n{essay}\n\n"
                "Grade this essay according to the rubric and reference material provided. "
                "Be specific about points deducted and explain why. "
                "First provide a detailed analysis, then summarize with total points deducted at the end."
            )
            
            # Base request parameters
            request_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert essay grader who provides detailed feedback."},
                    {"role": "user", "content": grading_request}
                ]
            }
            
            # Add vector store context if available
            if self.vector_store_id:
                request_params["tools"] = [{
                    "type": "retrieval",
                    "retrieval": {
                        "vector_store_ids": [self.vector_store_id]
                    }
                }]
            
            # Handle special cases for different model types
            if "o1" in model or "o3-mini" in model or "o4-mini" in model:
                # These models don't support temperature or might have restrictions
                pass
            else:
                # Add temperature and top_p for models that support them
                request_params["temperature"] = temperature
                request_params["top_p"] = top_p
            
            response = self.openai_client.chat.completions.create(**request_params)
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error grading with OpenAI model {model}: {str(e)}"
    
    def grade_essay_with_anthropic(self, model, prompt, rubric, reference, essay, temperature=0.7):
        """Grade an essay using an Anthropic model"""
        try:
            # Note: Anthropic doesn't support the same vector store retrieval as OpenAI,
            # so we include all context directly in the prompt
            grading_request = (
                f"Prompt:\n{prompt}\n\n"
                f"Reference Material:\n{reference}\n\n"
                f"Rubric:\n{rubric}\n\n"
                f"Essay to grade:\n{essay}\n\n"
                "Grade this essay according to the rubric and reference material provided. "
                "Be specific about points deducted and explain why. "
                "First provide a detailed analysis, then summarize with total points deducted at the end."
            )
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": grading_request}
                ],
                "temperature": temperature,
                "max_tokens": 4000
            }
            
            response = requests.post("https://api.anthropic.com/v1/messages", json=data, headers=headers)
            response.raise_for_status()
            
            return response.json()["content"][0]["text"]
            
        except Exception as e:
            return f"Error grading with Anthropic model {model}: {str(e)}"
    
    def grade_essay(self, provider, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        """Route the grading request to the appropriate API based on provider"""
        if provider == "OpenAI":
            return self.grade_essay_with_openai(model, prompt, rubric, reference, essay, temperature, top_p)
        elif provider == "Anthropic":
            return self.grade_essay_with_anthropic(model, prompt, rubric, reference, essay, temperature)
        else:
            return f"Unsupported provider: {provider}"


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


def main():
    st.set_page_config(page_title="Multi-Model Essay Grader", layout="wide")
    st.title("🤖 Multi-Model Essay Grader with Retrieval & Comparison")

    with st.expander("ℹ️ Instructions"):
        st.markdown("""
This tool grades essays using multiple AI models and compares outputs.
1. Upload your API Key, Prompt, Rubric, Reference, and Essays
2. Choose models and adjust temperature settings
3. Click "Grade Essays" to begin the process
4. Vector store context will improve grading consistency
5. Export results as Text, PDF, or ZIP
""")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        api_key_file = st.file_uploader("🔑 API Key (TXT)", type=["txt"])
    with col2:
        prompt_file = st.file_uploader("📝 Prompt (TXT)", type=["txt"])
    with col3:
        rubric_file = st.file_uploader("📋 Rubric (TXT)", type=["txt"])
    with col4:
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

    st.markdown("---")
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
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, 
                           help="Controls randomness: Lower values are more deterministic, higher values more creative.")
    
    top_p = st.slider("Top-p Sampling", 0.1, 1.0, 0.9, 0.01, 
                     help="Controls diversity: Lower values consider only the most likely tokens.")

    if st.button("🚀 Grade Essays"):
        if not api_key_file or not prompt_file or not rubric_file or not reference_file or not essay_files or not selected_models:
            st.error("Please upload all required files and select at least one model.")
        else:
            try:
                api_key = api_key_file.read().decode("utf-8").strip()
                prompt = prompt_file.read().decode("utf-8")
                rubric = rubric_file.read().decode("utf-8")
                reference = reference_file.read().decode("utf-8")
                
                # Initialize the model manager with the API key
                model_manager = ModelManager(api_key)
                
                # Create vector store for context (seamless to the user)
                with st.spinner("Preparing grading context..."):
                    model_manager.create_vector_store(prompt, rubric, reference)
                
                all_pdfs = []

                for i, essay_text in enumerate(essay_files):
                    essay_name = essay_names[i] if i < len(essay_names) else f"essay_{i+1}.txt"
                    
                    results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, model_info in enumerate(selected_models):
                        provider = model_info["provider"]
                        model = model_info["model"]
                        
                        status_text.text(f"Grading {essay_name} with {provider} {model}...")
                        
                        # Apply provider-specific settings
                        provider_config = MODEL_OPTIONS.get(provider, {})
                        model_temp = temperature
                        if provider_config.get("supports_temperature", False):
                            min_temp, max_temp = provider_config.get("temp_range", (0.0, 1.0))
                            model_temp = max(min_temp, min(temperature, max_temp))
                        
                        model_top_p = top_p
                        if not provider_config.get("supports_top_p", True):
                            model_top_p = None
                        
                        # Grade essay with the selected model
                        graded_content = model_manager.grade_essay(
                            provider, 
                            model, 
                            prompt, 
                            rubric, 
                            reference, 
                            essay_text,
                            temperature=model_temp,
                            top_p=model_top_p if model_top_p is not None else 0.9
                        )
                        
                        results[f"{provider} - {model}"] = graded_content
                        progress_bar.progress((idx + 1) / len(selected_models))
                        
                        # Add a small delay to avoid rate limits
                        time.sleep(0.5)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.subheader(f"📝 Results for {essay_name}")
                    display_comparison_table(results)
                    
                    # Save original essay content and graded results for download
                    original_and_graded = f"ORIGINAL ESSAY:\n\n{essay_text}\n\n" + "\n\n".join([f"=== GRADED BY {model} ===\n\n{content}" for model, content in results.items()])
                    st.download_button(
                        "📄 Download Text Results",
                        data=original_and_graded,
                        file_name=f"graded_{essay_name}.txt",
                        mime="text/plain",
                        key=f"text_{essay_name}"
                    )
                    
                    # Also provide PDF download
                    pdf = save_pdf(essay_name, results)
                    all_pdfs.append((essay_name, pdf))
                    st.download_button(
                        "📥 Download as PDF", 
                        data=pdf, 
                        file_name=f"graded_{essay_name}.pdf", 
                        mime="application/pdf",
                        key=f"pdf_{essay_name}"
                    )

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
