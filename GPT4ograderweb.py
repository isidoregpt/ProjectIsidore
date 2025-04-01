import streamlit as st
from openai import OpenAI
import datetime
import io
import zipfile

class Grader:
    def __init__(self):
        self.api_key = None
        self.client = None
        self.prompt = None
        self.rubric = None
        self.reference_material = None
        self.temperature = 1.0
        self.top_p = 0.5

    def set_api_key(self, api_key_content):
        self.api_key = api_key_content.strip()
        self.client = OpenAI(api_key=self.api_key)

    def set_prompt(self, prompt_content):
        self.prompt = prompt_content

    def set_rubric(self, rubric_content):
        self.rubric = rubric_content

    def set_reference_material(self, reference_material_content):
        self.reference_material = reference_material_content

    def grade(self, essay_files, iterations=2, temperature=1.0, top_p=0.5):
        self.temperature = temperature
        self.top_p = top_p

        if not all([self.api_key, self.prompt, self.rubric, self.reference_material, essay_files]):
            raise Exception("Please provide API key, prompt, rubric, reference material, and at least one essay.")

        graded_essays = {}
        for essay_file in essay_files:
            filename = essay_file.name
            original_essay_content = essay_file.read().decode("utf-8")
            essay_content = original_essay_content

            for _ in range(iterations):
                grading_request = (
                    "The following is an uncorrected student essay that contains grammatical and spelling mistakes. "
                    "Please correct the essay based on the provided rubric and reference material. "
                    "For each correction, place the corrected text in the 'Corrected Essay' section and indicate the mistakes "
                    "with point deductions in parentheses next to the mistake. Do not repeat or omit any part of the original essay. "
                    "Summarize the total points deducted at the end. "
                    "\n\nReference Material:\n{reference_material}\n\nPrompt:\n{prompt}\n\nStudent Essay (contains errors):\n{essay_content}\n\nRubric:\n{rubric}"
                ).format(
                    reference_material=self.reference_material,
                    prompt=self.prompt,
                    essay_content=essay_content,
                    rubric=self.rubric
                )

                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that grades and corrects essays."},
                        {"role": "user", "content": grading_request}
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                # Retrieve the generated text from the response pydantic model:
                essay_content = response.choices[0].message.content

            graded_content = f"Student Essay:\n\n{original_essay_content}\n\n{essay_content}"
            graded_filename = f"T_{self.temperature}_top_p_{self.top_p}_graded_{filename}"
            graded_essays[graded_filename] = graded_content

        return graded_essays

def main():
    # Use wide layout to maximize horizontal space
    st.set_page_config(page_title="Project Isidore GPT-4o", layout="wide")

    # "About" drop-down with project details
    with st.expander("About"):
        st.markdown(
            """
**Automated Essay Grader for Educators using GPT-4o with Custom Rubrics.**

Project Isidore GPT-4o Grader is a Python-based application designed to help educators experiment with AI-assisted grading. This tool uses OpenAI's GPT-4o models to automatically grade student essays in any language listed in the Languages.md file based on customizable grading rubrics. It provides automated feedback, identifies grammatical issues, incorrect word usage, and offers constructive corrections.

**Important:**  
Project Isidore is provided as an evaluation and testing tool for educators to explore the possibilities of AI in grading. This tool is not intended to replace traditional grading methods but rather to assist educators in experimenting with AI-driven feedback. Educators retain full control over final grades and should use the tool to supplement, not replace, human judgment.

**Acknowledgment:**  
This project was created with the help of ChatGPT-4 and GPT-4o, developed by OpenAI. These AI tools assisted in code generation, feature development, and documentation. However, all design choices, final decisions, and customizations were made by the author, Jonathan Graziola.

**Disclaimer:**  
**Consent, Anonymization, and Compliance with Legal Frameworks.**  
Project Isidore is designed for educational experimentation with AI-assisted grading. When using this tool, users must comply with applicable privacy laws such as the Family Educational Rights and Privacy Act (FERPA) in the United States and General Data Protection Regulation (GDPR) in Europe.

**Consent:**  
Ensure explicit consent from students, parents, or guardians (if applicable) before processing essays containing Personally Identifiable Information (PII).

**Anonymization:**  
We recommend anonymizing all student essays by removing or masking PII such as names, student IDs, and other identifying information.

**Legal Compliance:**  
Users are responsible for ensuring compliance with FERPA, GDPR, and other relevant laws regarding student data privacy.

**Handling PII:**  
This tool does not store or transmit any data beyond the grading session. However, users must understand OpenAI's privacy policies and how they apply to the data submitted.

**Examples of PII:**  
- Full names  
- Student ID numbers  
- Birth dates  
- Addresses or contact details

**Limitation of Liability:**  
The author of this tool, Jonathan Graziola, and contributors are not responsible for any misuse or violations of privacy laws arising from improper use of this application. Users are solely responsible for ensuring compliance with all applicable laws.

**Data Storage:**  
The app doesn't save any of your data to disk on a remote server. All files you upload are processed in memory during your session, and the graded output is packaged into an in‑memory ZIP file for you to download to your own computer.

The only exception is the data sent to the OpenAI API. That data is transmitted to OpenAI’s servers for processing, but it isn’t stored by the app on any external storage. Always check OpenAI’s data usage and retention policies for further details on how they handle transmitted content.
            """
        )

    st.title("Project Isidore GPT-4o")

    # Create horizontal columns for file uploads
    st.markdown("#### Upload Required Files")
    col1, col2, col3, col4, col5 = st.columns(5, gap="large")

    with col1:
        st.markdown("**API Key (TXT)**")
        api_key_file = st.file_uploader("", type=["txt"], key="api_key")
    with col2:
        st.markdown("**Student Prompt (TXT)**")
        prompt_file = st.file_uploader("", type=["txt"], key="prompt_file")
    with col3:
        st.markdown("**Rubric (TXT)**")
        rubric_file = st.file_uploader("", type=["txt"], key="rubric_file")
    with col4:
        st.markdown("**Reference Material (TXT)**")
        reference_file = st.file_uploader("", type=["txt"], key="reference_file")
    with col5:
        st.markdown("**Student Essays (TXT)**")
        essay_files = st.file_uploader("", type=["txt"], accept_multiple_files=True, key="essay_files")

    st.markdown("---")
    st.markdown("### Model Settings")
    colA, colB, colC = st.columns([1, 1, 1], gap="large")
    with colA:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
    with colB:
        top_p = st.slider("Nucleus Sampling (top_p)", min_value=0.1, max_value=0.99, value=0.5, step=0.01)
    with colC:
        iterations = st.number_input("Iterations", min_value=1, max_value=10, value=1, step=1)

    st.markdown("---")

    if st.button("Grade Essays"):
        if not api_key_file or not prompt_file or not rubric_file or not reference_file or not essay_files:
            st.error("Please upload all required files before grading.")
        else:
            try:
                grader = Grader()
                grader.set_api_key(api_key_file.read().decode("utf-8"))
                grader.set_prompt(prompt_file.read().decode("utf-8"))
                grader.set_rubric(rubric_file.read().decode("utf-8"))
                grader.set_reference_material(reference_file.read().decode("utf-8"))

                with st.spinner("Grading essays..."):
                    graded_essays = grader.grade(essay_files, iterations=iterations, temperature=temperature, top_p=top_p)

                st.success("Grading completed!")

                # Create an in-memory ZIP archive of the graded essays
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, content in graded_essays.items():
                        zip_file.writestr(filename, content)
                zip_buffer.seek(0)

                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = f"graded_essays_{current_time}.zip"

                st.download_button(
                    "Download All Graded Essays (ZIP)",
                    data=zip_buffer,
                    file_name=zip_filename,
                    mime="application/zip"
                )

                st.markdown("### Individual Graded Essays")
                for filename, content in graded_essays.items():
                    st.download_button(
                        f"Download {filename}",
                        data=content,
                        file_name=filename,
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
