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

    def grade(self, essay_files, iterations=1, temperature=0.10, top_p=0.90):
        self.temperature = temperature
        self.top_p = top_p

        if not all([self.api_key, self.prompt, self.rubric, self.reference_material, essay_files]):
            raise Exception("Please provide an API key, prompt, rubric, reference material, and at least one essay file.")

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
    st.set_page_config(page_title="Project Isidore GPT-4o", layout="wide")
    
    # Instructions Expander (on top)
    with st.expander("Instructions"):
        st.markdown(
            """
**Overview:**  
This app is an automated essay grader for educators using GPT-4o. It uses custom prompts, rubrics, and reference material to help grade student essays. The results can be downloaded individually or as a ZIP file.

**1. Upload Required Files:**  
- **API Key (TXT):**  
  *What it is:* A text file containing your OpenAI API key.  
  *Why you need it:* This key allows the app to access OpenAI’s GPT-4o model.  
  *How to get one:* Sign up at [OpenAI](https://platform.openai.com/signup) and create an API key from your account settings.  
- **Student Prompt (TXT):**  
  *What it is:* A text file with the prompt given to the student to create the essay.  
- **Rubric (TXT):**  
  *What it is:* A text file containing the grading criteria, including point deductions and corrections.  
- **Reference Material (TXT):**  
  *What it is:* Additional text that the student studied from or was used to educate the student on the subject matter.  
- **Student Essays Folder (TXT Files):**  
  *What it is:* **Select all the text files from the folder** containing all the student essays you want to grade.

**2. Model Settings:**  
- **Temperature:**  
  *What it is:* Controls the randomness of the AI’s responses.  
  *Why adjust it:* Lower values (e.g., 0.0–0.5) make the output more predictable, while higher values (e.g., 0.7–1.0) make it more creative.  
- **Nucleus Sampling (top_p):**  
  *What it is:* Another control for response randomness.  
  *Why adjust it:* Lower values restrict the range of possible outputs, and higher values allow more variety.  
- **Iterations:**  
  *What it is:* The number of times the grading process is run for each essay.  
  *Why adjust it:* More iterations can improve the quality of the final result but will take longer to process.

**3. Grade Essays Button:**  
After uploading all files and setting the parameters, click this button. The app will process each essay using the provided information and show a loading spinner until finished.

**4. Download Section:**  
- **Download All Graded Essays (ZIP):**  
  Download a single ZIP file containing all graded essays.  
- **Individual Graded Essays:**  
  Download each graded essay separately.
            """
        )
    
    # About Expander (below Instructions)
    with st.expander("Privacy and About"):
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
Project Isidore is designed for educational experimentation with AI-assisted grading. When using this tool, users must comply with applicable privacy laws such as FERPA in the United States and GDPR in Europe.

**Consent:**  
Ensure explicit consent from students, parents, or guardians (if applicable) before processing essays containing PII.

**Anonymization:**  
We recommend anonymizing all student essays by removing or masking PII such as names, student IDs, etc.

**Legal Compliance:**  
Users are responsible for ensuring compliance with FERPA, GDPR, and other relevant laws regarding student data privacy.

**Handling PII:**  
This tool does not save or transmit data beyond the grading session. However, check OpenAI’s policies for details on data handling.

**Limitation of Liability:**  
The author and contributors are not responsible for any misuse of this tool.

**Data Storage:**  
Files are processed in memory during your session, and the output is available for download. Only data sent to the OpenAI API is transmitted externally.
            """
        )
    
    st.title("Project Isidore GPT-4o")

    # Horizontal layout for file uploads
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
        st.markdown("**Student Essays Folder (TXT Files)**")
        essay_files = st.file_uploader(
            "Select all essay files from the folder",
            type=["txt"],
            accept_multiple_files=True,
            key="essay_files"
        )

    st.markdown("---")
    st.markdown("### Model Settings")
    colA, colB, colC = st.columns([1, 1, 1], gap="large")
    with colA:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.10, step=0.1)
    with colB:
        top_p = st.slider("Nucleus Sampling (top_p)", min_value=0.1, max_value=0.99, value=0.90, step=0.01)
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
