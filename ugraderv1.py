import streamlit as st
import datetime
import io
import zipfile
import os
import tempfile
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
        self.vector_store_created = False
        
    def create_vector_store(self, prompt, rubric, reference):
        """Create vector store for retrieval-augmented grading"""
        try:
            # Create a new vector store
            vs = self.openai_client.vector_stores.create(
                name="Essay Grading Context"
            )
            self.vector_store_id = vs.id
            
            # Save context files temporarily
            context_files = {
                "prompt.txt": prompt,
                "rubric.txt": rubric,
                "reference.txt": reference
            }
            
            # Use tempfile to safely handle encoding issues
            for name, content in context_files.items():
                with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", encoding="utf-8", delete=False) as f:
                    f.write(content)
                    temp_filename = f.name
                
                # Reopen in binary mode for upload
                with open(temp_filename, "rb") as f:
                    try:
                        # Use the upload_and_poll method to ensure file processing completes
                        self.openai_client.vector_stores.files.upload_and_poll(
                            vector_store_id=self.vector_store_id,
                            file=f
                        )
                    except Exception as upload_error:
                        st.warning(f"Error uploading {name}: {upload_error}")
                
                # Clean up temporary file
                try:
                    os.remove(temp_filename)
                except:
                    pass
                    
            self.vector_store_created = True
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            self.vector_store_created = False
            return False
    
    def get_vector_store_results(self, essay):
        """Perform a vector store search to get relevant context for grading"""
        try:
            if not self.vector_store_id:
                return None, False
            
            # Create a search query based on the essay content
            # Use only a small preview to avoid encoding issues
            preview = essay[:100] if len(essay) > 100 else essay
            search_query = f"What criteria from the rubric apply to this essay? {preview}"
            
            # Search the vector store
            results = self.openai_client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=search_query,
                max_num_results=5  # Limit to most relevant results
            )
            
            # Format results for inclusion in the prompt
            formatted_results = []
            for result in results.data:
                content_text = "\n".join([part.text for part in result.content])
                formatted_results.append(f"From {result.filename}:\n{content_text}")
            
            # Return both the formatted results and whether RAG was successful
            rag_success = len(formatted_results) > 0
            return "\n\n".join(formatted_results), rag_success
        except Exception as e:
            st.warning(f"Vector store search had an issue: {e}")
            return None, False
    
    def grade_essay_with_openai(self, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        """Grade an essay using an OpenAI model with retrieved context"""
        try:
            # Try to get relevant context from vector store
            vector_store_context, rag_success = self.get_vector_store_results(essay)
            
            # Prepare the grading request
            if vector_store_context:
                grading_request = (
                    f"Prompt:\n{prompt}\n\n"
                    f"Reference Material:\n{reference}\n\n"
                    f"Rubric:\n{rubric}\n\n"
                    f"Relevant Context:\n{vector_store_context}\n\n"
                    f"Essay to grade:\n{essay}\n\n"
                    "Grade this essay according to the rubric and reference material provided. "
                    "Be specific about points deducted and explain why. "
                    "First provide a detailed analysis, then summarize with total points deducted at the end."
                )
            else:
                # Fallback to standard prompt if vector store retrieval fails
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
            
            # Handle special cases for different model types
            if "o1" in model or "o3-mini" in model or "o4-mini" in model:
                # These models don't support temperature or might have restrictions
                pass
            else:
                # Add temperature and top_p for models that support them
                request_params["temperature"] = temperature
                request_params["top_p"] = top_p
            
            response = self.openai_client.chat.completions.create(**request_params)
            
            # Return content along with RAG status flags
            return {
                "content": response.choices[0].message.content,
                "vector_store_created": self.vector_store_created,
                "rag_success": rag_success
            }
            
        except Exception as e:
            return {
                "content": f"Error grading with OpenAI model {model}: {str(e)}",
                "vector_store_created": self.vector_store_created,
                "rag_success": False
            }
    
    def grade_essay_with_anthropic(self, model, prompt, rubric, reference, essay, temperature=0.7):
        """Grade an essay using an Anthropic model"""
        try:
            # Try to get relevant context from vector store
            vector_store_context, rag_success = self.get_vector_store_results(essay)
            
            # Prepare the grading request
            if vector_store_context:
                grading_request = (
                    f"Prompt:\n{prompt}\n\n"
                    f"Reference Material:\n{reference}\n\n"
                    f"Rubric:\n{rubric}\n\n"
                    f"Relevant Context:\n{vector_store_context}\n\n"
                    f"Essay to grade:\n{essay}\n\n"
                    "Grade this essay according to the rubric and reference material provided. "
                    "Be specific about points deducted and explain why. "
                    "First provide a detailed analysis, then summarize with total points deducted at the end."
                )
            else:
                # Fallback to standard prompt if vector store retrieval fails
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
            
            return {
                "content": response.json()["content"][0]["text"],
                "vector_store_created": self.vector_store_created,
                "rag_success": rag_success
            }
            
        except Exception as e:
            return {
                "content": f"Error grading with Anthropic model {model}: {str(e)}",
                "vector_store_created": self.vector_store_created,
                "rag_success": False
            }
    
    def grade_essay(self, provider, model, prompt, rubric, reference, essay, temperature=0.7, top_p=0.9):
        """Route the grading request to the appropriate API based on provider"""
        if provider == "OpenAI":
            return self.grade_essay_with_openai(model, prompt, rubric, reference, essay, temperature, top_p)
        elif provider == "Anthropic":
            return self.grade_essay_with_anthropic(model, prompt, rubric, reference, essay, temperature)
        else:
            return {
                "content": f"Unsupported provider: {provider}",
                "vector_store_created": False,
                "rag_success": False
            }


def save_pdf(essay_name, model_outputs):
    """Generate a PDF with grading results and RAG status"""
    try:
        # Create a temporary directory to store the PDF
        temp_dir = tempfile.mkdtemp()
        temp_pdf_path = os.path.join(temp_dir, f"graded_{essay_name}.pdf")
        
        # Create PDF with standard fonts only
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add first page and set standard font
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
            
        # Create a safe title
        safe_name = ''.join(c if ord(c) < 128 else '?' for c in essay_name)
        pdf.cell(200, 10, f"Graded Essay: {safe_name}", ln=True, align="C")
        
        # Process each model's output
        for model, output_data in model_outputs.items():
            # Set header font for model name
            pdf.set_font('Arial', 'B', 12)
                
            # Create safe model name with RAG status
            safe_model = ''.join(c if ord(c) < 128 else '?' for c in model)
            rag_status = (f"Vector Store: {'Success' if output_data['vector_store_created'] else 'Failed'}, "
                        f"RAG: {'Success' if output_data['rag_success'] else 'Failed'}")
            
            pdf.cell(200, 10, f"Model: {safe_model}", ln=True)
            pdf.cell(200, 10, f"RAG Status: {rag_status}", ln=True)
            
            # Set body font for content
            pdf.set_font('Arial', '', 10)
                
            # Process text line by line with encoding safety
            for line in output_data["content"].split("\n"):
                # Replace problematic characters
                safe_line = ''.join(c if ord(c) < 128 else '?' for c in line)
                pdf.multi_cell(0, 5, safe_line)
            pdf.ln()

        # Output to temporary file
        pdf.output(temp_pdf_path)
        
        # Read the file into memory
        with open(temp_pdf_path, 'rb') as f:
            pdf_bytes = f.read()
            
        # Clean up the temporary directory
        try:
            os.remove(temp_pdf_path)
            os.rmdir(temp_dir)
        except:
            pass
            
        return pdf_bytes
        
    except Exception as e:
        st.warning(f"PDF generation failed: {e}. Using text format instead.")
        # If PDF generation fails, return None and we'll handle it in the calling function
        return None


def display_comparison_table(results):
    """Display results table and return dataframe for export"""
    try:
        total_scores = []
        rows = []
        
        for model, output_data in results.items():
            content = output_data["content"]
            vector_store_created = output_data["vector_store_created"]
            rag_success = output_data["rag_success"]
            
            # Extract points deducted and summary
            lines = content.splitlines()
            deduction_line = next((line for line in lines if "points deducted" in line.lower()), "Not Found")
            summary = next((line for line in lines[::-1] if line.strip()), "")
            
            try:
                # Extract numerical value safely
                digits = ''.join(c for c in deduction_line if c.isdigit() or c == '.')
                points = float(digits) if digits else 0
            except:
                points = 0
                
            total_scores.append((model, points))
            
            # Add RAG status to the table
            rag_status = (f"Vector Store: {'Success' if vector_store_created else 'Failed'}, "
                          f"RAG: {'Success' if rag_success else 'Failed'}")
            
            rows.append({
                "Model": model, 
                "RAG Status": rag_status,
                "Points Deducted": deduction_line, 
                "Summary": summary[:150]
            })
            
        # Create and display the dataframe
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        
        if total_scores:
            st.subheader("📊 Average Points Deducted")
            score_df = pd.DataFrame(total_scores, columns=["Model", "Points Deducted"])
            avg = score_df.groupby("Model")["Points Deducted"].mean().reset_index()
            st.table(avg)
            
        # Return dataframe for CSV export
        return df
    except Exception as e:
        st.error(f"Error displaying comparison table: {e}")
        return None


def main():
    st.set_page_config(page_title="Multi-Model Essay Grader", layout="wide")
    st.title("🤖 Multi-Model Essay Grader with Vector Search")

    with st.expander("ℹ️ Instructions"):
        st.markdown("""
This tool grades essays using multiple AI models and compares outputs.
1. Upload your API Key, Prompt, Rubric, Reference, and Essays
2. Choose models and adjust temperature settings
3. Click "Grade Essays" to begin the process
4. Vector store retrieval enhances grading consistency
5. Export results as a ZIP file containing TXT, PDF, and CSV formats
6. RAG Status shows whether Vector Store and Retrieval worked for each model
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
        try:
            with zipfile.ZipFile(essay_zip) as z:
                essay_names = [f for f in z.namelist() if f.endswith(".txt")]
                # Use 'replace' for decoding errors
                essay_files = [z.read(f).decode("utf-8", errors="replace") for f in essay_names]
        except Exception as e:
            st.error(f"Error reading ZIP file: {e}")
    else:
        uploaded_files = st.file_uploader("📂 Upload one or more student essays", type=["txt"], accept_multiple_files=True)
        if uploaded_files:
            essay_names = [f.name for f in uploaded_files]
            essay_files = []
            for f in uploaded_files:
                try:
                    # Use 'replace' for decoding errors
                    essay_files.append(f.read().decode("utf-8", errors="replace"))
                except Exception as e:
                    st.error(f"Error reading file {f.name}: {e}")
                    essay_files.append(f"[Error reading file: {str(e)}]")

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
                # Read files with error handling
                try:
                    api_key = api_key_file.read().decode("utf-8", errors="replace").strip()
                    prompt = prompt_file.read().decode("utf-8", errors="replace")
                    rubric = rubric_file.read().decode("utf-8", errors="replace")
                    reference = reference_file.read().decode("utf-8", errors="replace")
                except Exception as e:
                    st.error(f"Error reading input files: {e}")
                    return
                
                # Initialize the model manager with the API key
                model_manager = ModelManager(api_key)
                
                # Create vector store for semantic search
                with st.spinner("Creating vector store for semantic search..."):
                    vector_store_created = model_manager.create_vector_store(prompt, rubric, reference)
                    if vector_store_created:
                        st.success("Vector store created successfully.")
                    else:
                        st.warning("Vector store creation skipped - continuing with standard grading.")
                
                # Lists to hold all the generated files
                all_essay_data = []

                for i, essay_text in enumerate(essay_files):
                    if i >= len(essay_names):
                        essay_name = f"essay_{i+1}.txt"
                    else:
                        essay_name = essay_names[i]
                    
                    results = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create a persistent container for RAG status that will remain
                    # even after all models have finished processing
                    rag_status_container = st.container()
                    with rag_status_container:
                        st.subheader("🔍 RAG Status Tracking")
                        rag_status_table = st.empty()
                    
                    # Track RAG status for all models
                    rag_statuses = []
                    
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
                        try:
                            result = model_manager.grade_essay(
                                provider, 
                                model, 
                                prompt, 
                                rubric, 
                                reference, 
                                essay_text,
                                temperature=model_temp,
                                top_p=model_top_p if model_top_p is not None else 0.9
                            )
                            
                            # Store results with RAG status flags
                            results[f"{provider} - {model}"] = result
                            
                            # Update RAG status for this model
                            rag_statuses.append({
                                "Model": f"{provider} - {model}",
                                "Vector Store": "Success" if result["vector_store_created"] else "Failed",
                                "RAG Success": "Success" if result["rag_success"] else "Failed"
                            })
                            
                            # Update the RAG status table as we go
                            rag_status_table.dataframe(pd.DataFrame(rag_statuses))
                                
                        except Exception as e:
                            results[f"{provider} - {model}"] = {
                                "content": f"Error grading: {str(e)}",
                                "vector_store_created": model_manager.vector_store_created,
                                "rag_success": False
                            }
                            
                            rag_statuses.append({
                                "Model": f"{provider} - {model}",
                                "Vector Store": "Success" if model_manager.vector_store_created else "Failed",
                                "RAG Success": "Failed"
                            })
                            
                            rag_status_table.dataframe(pd.DataFrame(rag_statuses))
                            st.error(f"Error grading with {provider} {model}: {e}")
                        
                        progress_bar.progress((idx + 1) / len(selected_models))
                        
                        # Add a small delay to avoid rate limits
                        time.sleep(0.5)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.subheader(f"📝 Results for {essay_name}")
                    results_df = display_comparison_table(results)
                    
                    # Prepare the files for this essay
                    essay_files_data = {}
                    
                    # 1. Generate text file
                    text_content = f"ORIGINAL ESSAY:\n\n{essay_text}\n\n" + "\n\n".join([
                        f"=== GRADED BY {model} ===\n"
                        f"Vector Store: {'Success' if data['vector_store_created'] else 'Failed'}, "
                        f"RAG: {'Success' if data['rag_success'] else 'Failed'}\n\n"
                        f"{data['content']}" 
                        for model, data in results.items()
                    ])
                    essay_files_data["txt"] = text_content.encode('utf-8')
                    
                    # 2. Generate PDF file
                    pdf_bytes = save_pdf(essay_name, results)
                    if pdf_bytes:
                        essay_files_data["pdf"] = pdf_bytes
                    
                    # 3. Generate CSV file if dataframe is available
                    if results_df is not None:
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        essay_files_data["csv"] = csv_buffer.getvalue().encode('utf-8')
                    
                    # Store the data for this essay
                    all_essay_data.append((essay_name, essay_files_data))
                    
                    # Create a ZIP for this essay
                    essay_zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(essay_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                        zipf.writestr(f"graded_{essay_name}.txt", essay_files_data["txt"])
                        if "pdf" in essay_files_data:
                            zipf.writestr(f"graded_{essay_name}.pdf", essay_files_data["pdf"])
                        if "csv" in essay_files_data:
                            zipf.writestr(f"graded_{essay_name}_results.csv", essay_files_data["csv"])
                    
                    essay_zip_buffer.seek(0)
                    st.download_button(
                        "📦 Download All Formats",
                        data=essay_zip_buffer,
                        file_name=f"graded_{essay_name}_all.zip",
                        mime="application/zip",
                        key=f"zip_{essay_name}"
                    )

                # Create a single ZIP file with all essays if multiple essays exist
                if len(essay_files) > 1:
                    try:
                        all_essays_zip = io.BytesIO()
                        with zipfile.ZipFile(all_essays_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
                            for essay_name, files_data in all_essay_data:
                                # Add all file formats for each essay
                                zipf.writestr(f"graded_{essay_name}.txt", files_data["txt"])
                                if "pdf" in files_data:
                                    zipf.writestr(f"graded_{essay_name}.pdf", files_data["pdf"])
                                if "csv" in files_data:
                                    zipf.writestr(f"graded_{essay_name}_results.csv", files_data["csv"])
                                
                        all_essays_zip.seek(0)
                        st.download_button(
                            "📦 Download ALL Essays (All Formats)",
                            data=all_essays_zip, 
                            file_name="all_graded_essays.zip", 
                            mime="application/zip"
                        )
                    except Exception as e:
                        st.error(f"Error creating combined ZIP file: {e}")

            except Exception as e:
                st.error(f"❌ Error grading essays: {e}")

if __name__ == "__main__":
    main()
