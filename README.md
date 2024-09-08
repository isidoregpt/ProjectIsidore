# Project Isidore GPT-4 & GPT-4o Grader

**Automated Essay Grader for Educators Using GPT-4 and GPT-4o with Custom Rubrics**

**Project Isidore GPT-4 & GPT-4o Grader** is a Python-based application designed to help educators experiment with AI-assisted grading. This tool uses OpenAI's GPT-4 or GPT-4o models to automatically grade student essays in any language listed in the Languages.md file based on customizable grading rubrics. It provides **automated feedback**, identifies grammatical issues, incorrect word usage, and offers constructive corrections.

### **Important: Intended for Educational Evaluation and Testing**

Project Isidore is provided as an **evaluation and testing tool** for educators to explore the possibilities of AI in grading. This tool is **not intended to replace traditional grading methods** but rather to assist educators in experimenting with AI-driven feedback. **Educators retain full control over final grades** and should use the tool to supplement, not replace, human judgment.

----------

### **Acknowledgment**

This project was created with the help of ChatGPT-4 and ChatGPT-4o, developed by OpenAI. These AI tools assisted in code generation, feature development, and documentation. However, all design choices, final decisions, and customizations were made by the author, **Jonathan Graziola**.

----------

### **Disclaimer: Consent, Anonymization, and Compliance with Legal Frameworks**

**Project Isidore** is designed for **educational experimentation** with AI-assisted grading. When using this tool, users must comply with applicable privacy laws such as the **Family Educational Rights and Privacy Act (FERPA)** in the United States and **General Data Protection Regulation (GDPR)** in Europe.

#### **Key Considerations**:

1.  **Consent**: Ensure explicit consent from students, parents, or guardians (if applicable) before processing essays containing Personally Identifiable Information (PII).
2.  **Anonymization**: We recommend anonymizing all student essays by removing or masking PII such as names, student IDs, and other identifying information.
3.  **Legal Compliance**: Users are responsible for ensuring compliance with FERPA, GDPR, and other relevant laws regarding student data privacy.
4.  **Handling PII**: This tool does not store or transmit any data beyond the grading session. However, users must understand OpenAI's privacy policies and how they apply to the data submitted.

#### **Examples of PII**:

-   Full names
-   Student ID numbers
-   Birth dates
-   Addresses or contact details

**Limitation of Liability**: The author of this tool, Jonathan Graziola, and contributors are not responsible for any misuse or violations of privacy laws arising from improper use of this application. Users are solely responsible for ensuring compliance with all applicable laws.

----------

### **Key Features**

1.  **Custom Rubric Integration**  
    Project Isidore allows users to load custom grading rubrics to personalize grading criteria. The rubric is a simple text file that outlines point deductions for various writing issues like grammar, spelling, or punctuation.
    
    **Example Rubric**:
    
    -   Grammar: -1 point per error
    -   Spelling: -0.5 points per mistake
    -   Punctuation: -0.5 points per mistake
    -   Verb tense: -1 point for incorrect usage
    -   Incomplete sentences: -2 points
2.  **AI-Powered Essay Correction**  
    The tool leverages GPT-4 or GPT-4o to identify mistakes and provide automatic corrections based on your custom rubric. It offers clear and concise feedback, improving student writing through automated suggestions.
    
    **Example**:
    
    -   "I has a apple" corrected to "I have an apple" with deductions for verb tense and grammar.
3.  **Configurable Model Settings**  
    Users can fine-tune GPT-4 or GPT-4o’s **temperature** and **nucleus sampling (top_p)** settings to control the model’s behavior.
    
    -   **Low temperature (e.g., 0.2)**: More deterministic output.
    -   **High temperature (e.g., 0.8)**: Introduces more creativity in the model’s responses.
    -   **Low top_p (e.g., 0.1)**: Restricts the model to more predictable outputs.
    -   **High top_p (e.g., 0.9)**: Allows for more diverse suggestions.
4.  **Automated Batch Grading**  
    The tool processes multiple essays in a single batch. Corrected essays are saved with filenames that reflect the model settings (temperature and top_p) used during grading.
    
    **Example Filenames**:
    
    -   `T_0.2_top_p_0.1_graded_student1.txt`
    -   `T_0.8_top_p_0.9_graded_student2.txt`
5.  **Detailed Feedback**  
    Each essay is evaluated with detailed feedback on every mistake, along with explanations and point deductions based on your rubric. Both the original and corrected versions of the essays are provided for review.
    
    **Example Feedback**:
    
    -   Original: "I has a apple"
    -   Corrected: "I have an apple"
    -   Feedback: Verb conjugation error (-1 point); article use error (-0.5 points).

----------

### **How It Works**

1.  **Load API Key**:  
    Input your OpenAI API key (for GPT-4 or GPT-4o) to begin.
2.  **Select Files**:  
    Load your custom rubric and the folder of student essays to grade.
3.  **Configure Settings**:  
    Adjust temperature and top_p for fine-tuned grading.
4.  **Grade**:  
    Click 'Grade' to process all essays in the folder. The system will automatically generate feedback and corrections based on your rubric.

----------

### **Requirements**

-   OpenAI API Key (GPT-4 or GPT-4o access) https://platform.openai.com/docs/quickstart
-   Python 3.x https://www.python.org/downloads/
-   Tkinter (for the graphical user interface) https://docs.python.org/3/library/tkinter.html

----------

### **Running the Application (PC and Mac)**

**1. Download and Install Python**:

-   For Windows, download Python from the official website, check "Add Python to PATH," and install.
-   For macOS, download and install Python, then verify using `python3 --version`.

**2. Download Project Isidore**:

1.  Download the ZIP file from GitHub.
2.  Extract the files to your preferred folder.

**3. Install Required Libraries**:

-   **Windows**:  
    Open Command Prompt and navigate to the project folder. Run `pip install openai tkinter`.
-   **macOS**:  
    Open Terminal, navigate to the project folder, and run `pip3 install openai tk`.

**4. Get Your OpenAI API Key**:  
Obtain your API key from OpenAI's API section and save it as a text file (e.g., apikey.txt).

**5. Run the Application**:  
In your terminal or command prompt, run the appropriate Python command:

-   **Windows**: `python isidore_grader.py`
-   **macOS**: `python3 isidore_grader.py`

**6. Start Grading**:  
Enter your API key, select the student essays and rubric, configure the settings, and start the grading process.

----------

### **License**

This project is licensed under the **GNU General Public License v3.0**.  
For more details, see: [https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html)

----------

### **Author**: Jonathan Graziola (isidore.gpt@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **WITHOUT ANY WARRANTY**; without even the implied warranty of **MERCHANTABILITY** or **FITNESS FOR A PARTICULAR PURPOSE**. See the GNU General Public License for more details.
