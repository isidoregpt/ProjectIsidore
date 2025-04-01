import streamlit as st
import openai
import datetime
import io
import zipfile

class Grader:
    def __init__(self):
        self.api_key = None
        self.prompt = None
        self.rubric = None
        self.reference_material = None
        self.temperature = 1.0
        self.top_p = 0.5

    def set_api_key(self, api_key_content):
        self.api_key = api_key_content.strip()
        openai.api_key = self.api_key

    def set_prompt(self, prompt_content):
        self.prompt = prompt_content

    def set_rubric(self, rubric_content):
        self.rubric = rubric_content

    def set_reference_material(self, reference_material_content):
        self.reference_material = reference_material_content

    def grade(self, essay_files, iterations=2, temperature=1.0, top_p=0.5):
        self.temperature = temperature
        self.top_p = top_p
        # Check that all necessary data is loaded
        if not all([self.api_key, self.prompt, self.rubric, self.reference_material, essay_files]):
            raise Exception("Ensure that API key, prompt, rubric, reference material, and at least one essay are uploaded.")
        
        graded_essays = {}
        for essay_file in essay_files:
            filename = essay_file.name
            # Read the entire content of the uploaded file
            original_essay_content = essay_file.read().decode("utf-8")
            essay_content = original_essay_content

            for _ in range(iterations):
                grading_request = (
                    "The following is an uncorrected student essay that contains grammatical and spelling mistakes. "
                    "Please correct the essay based on the provided rubric and reference material. "
                    "For each correction, place the corrected text in the 'Corrected Essay' section and indicate the mistakes with point deductions in parentheses next to the mistake. "
                    "Do not repeat or omit any part of the original essay. Summarize the total points deducted at the end. "
                    "\n\nReference Material:\n{reference_material}\n\nPrompt:\n{prompt}\n\nStudent Essay (contains errors):\n{essay_content}\n\nRubric:\n{rubric}"
                ).format(
                    reference_material=self.reference_material,
                    prompt=self.prompt,
                    essay_content=essay_content,
                    rubric=self.rubric
                )
                
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that grades and corrects essays."},
                        {"role": "user", "content": grading_request}
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                essay_content = response['choices'][0]['message']['content']

            graded_content = f"Student Essay:\n\n{original_essay_content}\n\n{essay_content}"
            graded_filename = f"T_{self.temperature}_top_p_{self.top_p}_graded_{filename}"
            graded_essays[graded_filename] = graded_content

        return graded_essays

def main():
    st.title("Project Isidore GPT-4o")

    st.markdown("### Upload Required Files")
    api_key_file = st.file_uploader("API Key (TXT file)", type=["txt"])
    prompt_file = st.file_uploader("Student Prompt (TXT file)", type=["txt"])
    rubric_file = st.file_uploader("Rubric (TXT file)", type=["txt"])
    reference_file = st.file_uploader("Reference Material (TXT file)", type=["txt"])
    essay_files = st.file_uploader("Student Essays (Select one or more TXT files)", type=["txt"], accept_multiple_files=True)

    st.markdown("### Model Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
    top_p = st.slider("Nucleus Sampling (top_p)", min_value=0.1, max_value=0.99, value=0.5, step=0.01)
    iterations = st.number_input("Iterations", min_value=1, max_value=10, value=2, step=1)

    if st.button("Grade Essays"):
        if not api_key_file or not prompt_file or not rubric_file or not reference_file or not essay_files:
            st.error("Please upload all required files before grading.")
        else:
            try:
                grader = Grader()
                # Read and set file contents
                grader.set_api_key(api_key_file.read().decode("utf-8"))
                grader.set_prompt(prompt_file.read().decode("utf-8"))
                grader.set_rubric(rubric_file.read().decode("utf-8"))
                grader.set_reference_material(reference_file.read().decode("utf-8"))
                
                with st.spinner("Grading essays..."):
                    graded_essays = grader.grade(essay_files, iterations=iterations, temperature=temperature, top_p=top_p)
                
                st.success("Grading completed!")

                # Create a ZIP archive in memory containing all graded essays
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, content in graded_essays.items():
                        zip_file.writestr(filename, content)
                zip_buffer.seek(0)
                
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = f"graded_essays_{current_time}.zip"
                st.download_button("Download All Graded Essays (ZIP)", data=zip_buffer, file_name=zip_filename, mime="application/zip")
                
                # Optionally, provide individual download buttons for each graded file.
                st.markdown("### Individual Graded Essays")
                for filename, content in graded_essays.items():
                    st.download_button(f"Download {filename}", data=content, file_name=filename, mime="text/plain")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
