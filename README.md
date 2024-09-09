## Project Isidore GPT-4 & GPT-4o Grader

Automated Essay Grader for Educators Using GPT-4 and GPT-4o with Custom Rubrics

Project Isidore GPT-4 & GPT-4o Grader is a Python-based application designed to help educators experiment with AI-assisted grading. This tool uses OpenAI's GPT-4 or GPT-4o models to automatically grade student essays in any language listed in the `Languages.md` file based on customizable grading rubrics. It provides automated feedback, identifies grammatical issues, incorrect word usage, and offers constructive corrections.

### Important: Intended for Educational Evaluation and Testing

Project Isidore is provided as an evaluation and testing tool for educators to explore the possibilities of AI in grading. This tool is not intended to replace traditional grading methods but rather to assist educators in experimenting with AI-driven feedback. Educators retain full control over final grades and should use the tool to supplement, not replace, human judgment.

### Acknowledgment

This project was created with the help of ChatGPT-4 and GPT-4o, developed by OpenAI. These AI tools assisted in code generation, feature development, and documentation. However, all design choices, final decisions, and customizations were made by the author, Jonathan Graziola.

### Disclaimer: Consent, Anonymization, and Compliance with Legal Frameworks

Project Isidore is designed for educational experimentation with AI-assisted grading. When using this tool, users must comply with applicable privacy laws such as the Family Educational Rights and Privacy Act (FERPA) in the United States and General Data Protection Regulation (GDPR) in Europe.

#### Key Considerations:

-   **Consent:** Ensure explicit consent from students, parents, or guardians (if applicable) before processing essays containing Personally Identifiable Information (PII).
-   **Anonymization:** We recommend anonymizing all student essays by removing or masking PII such as names, student IDs, and other identifying information.
-   **Legal Compliance:** Users are responsible for ensuring compliance with FERPA, GDPR, and other relevant laws regarding student data privacy.
-   **Handling PII:** This tool does not store or transmit any data beyond the grading session. However, users must understand OpenAI's privacy policies and how they apply to the data submitted.

### Examples of PII:

-   Full names
-   Student ID numbers
-   Birth dates
-   Addresses or contact details

**Limitation of Liability:** The author of this tool, Jonathan Graziola, and contributors are not responsible for any misuse or violations of privacy laws arising from improper use of this application. Users are solely responsible for ensuring compliance with all applicable laws.

----------

### Key Features

#### Custom Rubric Integration

Project Isidore allows users to load custom grading rubrics to personalize grading criteria. The rubric is a simple text file that outlines point deductions for various writing issues like grammar, spelling, or punctuation.

**Example Rubric:**

-   Grammar: -1 point per error
-   Spelling: -0.5 points per mistake
-   Punctuation: -0.5 points per mistake
-   Verb tense: -1 point for incorrect usage
-   Incomplete sentences: -2 points

#### AI-Powered Essay Correction

The tool leverages GPT-4 or GPT-4o to identify mistakes and provide automatic corrections based on your custom rubric. It offers clear and concise feedback, improving student writing through automated suggestions.

**Example:**

> "I has a apple" corrected to "I have an apple" with deductions for verb tense and grammar.

#### Configurable Model Settings

Users can fine-tune GPT-4 or GPT-4o’s temperature and nucleus sampling (top_p) settings to control the model’s behavior.

-   **Low temperature (e.g., 0.2):** More deterministic output.
-   **High temperature (e.g., 0.8):** Introduces more creativity in the model’s responses.
-   **Low top_p (e.g., 0.1):** Restricts the model to more predictable outputs.
-   **High top_p (e.g., 0.9):** Allows for more diverse suggestions.

#### Automated Batch Grading

The tool processes multiple essays in a single batch. Corrected essays are saved with filenames that reflect the model settings (temperature and top_p) used during grading.

**Example Filenames:**

-   `T_0.2_top_p_0.1_graded_student1.txt`
-   `T_0.8_top_p_0.9_graded_student2.txt`

#### Detailed Feedback

Each essay is evaluated with detailed feedback on every mistake, along with explanations and point deductions based on your rubric. Both the original and corrected versions of the essays are provided for review.

**Example Feedback:**

-   **Original:** "I has a apple"
-   **Corrected:** "I have an apple"
-   **Feedback:** Verb conjugation error (-1 point); article use error (-0.5 points).

----------

### Student Prompt and Reference Material

In addition to grading based on a custom rubric, **Project Isidore** allows educators to load **Student Prompts** and **Reference Materials** for more contextual and accurate grading.

#### Student Prompt

Educators can upload a text file that contains the **original essay prompt** given to students. The prompt is used by the AI models to ensure that corrections and feedback are provided in relation to the specific instructions of the assignment. The prompt is treated as part of the grading context, so the AI can assess if the student answered the question or task correctly.

**How to Load the Student Prompt:**

1.  Click the "Student Prompt" button in the application.
2.  Select the text file containing the prompt.
3.  The application will load the prompt, which will be referenced when grading the student essays.

#### Reference Material

If an assignment requires students to reference external materials (e.g., articles, textbooks, or documents), educators can upload these as **Reference Material**. This helps the AI ensure that students are correctly using or citing the required sources in their essays.

**How to Load the Reference Material:**

1.  Click the "Reference Material" button in the application.
2.  Select the text file containing the reference material (e.g., PDF converted to text, textbook excerpt).
3.  The AI will use this information to check for proper use of references, quotations, or concepts from the material.

----------

### How It Works

1.  **Load API Key:** Input your OpenAI API key (for GPT-4 or GPT-4o) to begin.
2.  **Select Files:** Load the custom rubric, the folder of student essays, the student prompt, and any reference material.
3.  **Configure Settings:** Adjust temperature and top_p for fine-tuned grading.
4.  **Grade:** Click 'Grade' to process all essays. The system will automatically generate feedback and corrections based on the rubric, prompt, and reference material.

----------

### Requirements

-   **OpenAI API Key (GPT-4 or GPT-4o access)**: [OpenAI Quickstart Guide](https://platform.openai.com/docs/quickstart)
-   **Python 3.x**: [Download Python](https://www.python.org/downloads/)
-   **Tkinter (for the graphical user interface)**: [Tkinter Documentation](https://docs.python.org/3/library/tkinter.html)

----------

### Running the Application (PC and Mac)

1.  **Download and Install Python**:
    
    -   For Windows, download Python from the official website, check "Add Python to PATH," and install.
    -   For macOS, download and install Python, then verify using `python3 --version`.
2.  **Download Project Isidore**:
    
    -   Download the ZIP file from GitHub.
    -   Extract the files to your preferred folder.
3.  **Install Required Libraries**:
    
    -   Windows:
        
        bash
        
        Copy code
        
        `pip install openai tkinter` 
        
    -   macOS:
        
        bash
        
        Copy code
        
        `pip3 install openai tk` 
        
4.  **Get Your OpenAI API Key**:
    
    -   Obtain your API key from OpenAI's API section and save it as a text file (e.g., `apikey.txt`).
5.  **Run the Application**:
    
    -   In your terminal or command prompt, run the appropriate Python command:
        -   Windows: `python isidore_grader.py`
        -   macOS: `python3 isidore_grader.py`
6.  **Start Grading**:
    
    -   Enter your API key, select the student essays, rubric, prompt, reference material, configure the settings, and start the grading process.

----------

### License

This project is licensed under the GNU General Public License v3.0.  
For more details, see: [GNU License](https://www.gnu.org/licenses/gpl-3.0.en.html)
