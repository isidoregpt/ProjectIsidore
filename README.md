Project Isidore GPT-4 & GPT-4o Grader

Automated Essay Grader Using GPT-4 and GPT-4o with Custom Rubrics

Project Isidore GPT-4 & GPT-4o Grader is a Python application that uses OpenAI's GPT-4 or GPT-4o to automatically grade student essays based on customizable grading rubrics. This tool streamlines grading for educators by identifying grammatical errors, improper word usage, and other writing issues while providing constructive corrections. You can load your own grading rubric and student essays to get detailed feedback and point deductions for specific mistakes.

Acknowledgment

This project was written with the assistance of ChatGPT-4 and ChatGPT-4o, developed by OpenAI. These AI tools helped generate code, refine features, and structure documentation to make the project more accessible and efficient.

While AI was a significant part of the development process, all decisions, design choices, and customizations were made by the author, Jonathan Graziola.

Disclaimer: Consent, Anonymization, and Compliance with Legal Frameworks

Important: This project is designed to assist with automated grading of essays using GPT-4 and GPT-4o. When using this tool, it is crucial to ensure compliance with applicable privacy laws and regulations, such as the Family Educational Rights and Privacy Act (FERPA) in the United States.

Key Considerations:

Consent: Ensure you have explicit consent from students, parents, or guardians (where applicable) before processing essays that contain personally identifiable information (PII). This tool is not designed to handle sensitive student data without proper consent.

Anonymization: To protect privacy, it is recommended that any essays processed through this tool be anonymized. Remove or mask PII such as names, student ID numbers, and any other identifying information before submission.

Compliance with FERPA and Other Privacy Laws: If you are using this tool in an educational setting in the United States, ensure you are compliant with FERPA, which protects the privacy of student education records. For other countries, comply with relevant data protection laws (e.g., GDPR in Europe).

Handling PII: This tool does not store or transmit any data beyond the grading session, but users are responsible for ensuring the proper handling and protection of PII when using OpenAI's services. Ensure you understand OpenAI's privacy policy and how it applies to the data you provide.

Examples of PII:

Full names
Student ID numbers
Birth dates
Addresses or contact details
Best Practice: Before using this tool, ensure that any identifiable information is removed from the essays. For example, replace full names with "Student A" or other pseudonyms, and avoid submitting any sensitive information through the system.

Limitation of Liability:

The author of this tool, Jonathan Graziola, and the contributors are not responsible for the misuse of this application or any violations of privacy laws that may arise from improper use. Users of this tool are solely responsible for ensuring compliance with applicable laws and for obtaining all necessary consents.

Key Features:

Custom Rubric Integration:
Load your own grading rubric to personalize grading standards. The rubric is a simple text file that outlines the grading rules you want the system to follow. For example, you can specify how many points to deduct for incorrect grammar, spelling, or punctuation.

Example Rubric:

Grammar: -1 point for each error
Spelling: -0.5 points for each mistake
Punctuation: -0.5 points for each mistake
Verb tense: -1 point for each incorrect use
Wrong word: -1 point for each instance
Incomplete sentences: -2 points
This rubric will be used by the system to evaluate each essay. You can customize the rubric file to fit your needs.

AI-Powered Essay Correction:
The grader uses either GPT-4 or GPT-4o to identify mistakes and provide corrections. It offers clear, concise feedback to help improve student writing. The corrections are based on your rubric, ensuring that each essay is evaluated according to your specific guidelines.

How it works:

Grammar mistakes are highlighted and corrected according to the rubric.
Word usage errors are pointed out, with suggestions for better alternatives.
Spelling mistakes are automatically fixed, with point deductions if specified in the rubric.
Example: A sentence like "I has a apple" would be corrected to "I have an apple," with corresponding deductions for verb tense and grammar.

Configurable Model Settings:
You can adjust GPT-4 or GPT-4o’s temperature and nucleus sampling (top_p) settings to fine-tune the model’s creativity and determinism.

Temperature:

Controls how deterministic the model is.

Low Temperature (e.g., 0.2): Makes the output more deterministic and focused on repeating predictable corrections.
Example: "You should use 'have' instead of 'has'."
High Temperature (e.g., 0.8): Introduces more randomness, potentially offering more creative solutions or alternative phrasings.
Example: "It would be better to say 'I have an apple,' as 'has' is incorrect in this context."
Nucleus Sampling (top_p):

Controls the diversity of responses by limiting the model to consider only the most likely words until a certain probability threshold is reached.

Low top_p (e.g., 0.1): Makes the output more predictable.
Example: "You should say 'I have an apple'."
High top_p (e.g., 0.9): Allows for more varied and creative corrections.
Example: "It's more appropriate to say 'I've got an apple' or 'I have an apple'."
Tip: For a more deterministic and focused grading process, set a low temperature (e.g., 0.2) and low top_p (e.g., 0.1). For more creative feedback, set a higher temperature (e.g., 0.8) and higher top_p (e.g., 0.9).

Automated Grading for Batch Essays:
The grader can process multiple student essays from a folder in one batch. Simply load a folder containing multiple student essay files, and the system will process each one according to your rubric.

Each corrected essay will be saved with a custom filename that includes the temperature and top_p settings used during grading. This allows you to easily track how different model settings affect the output.

Example Custom Filenames:

T_0.2_top_p_0.1_graded_student1.txt
T_0.8_top_p_0.9_graded_student2.txt
In these examples, the filenames include the temperature (T) and nucleus sampling (top_p) settings, followed by the student's original filename. This helps you compare how different settings impact the grading process.

Detailed Feedback:
The grader generates detailed feedback for each essay, including corrections and explanations for the point deductions based on your rubric. You'll get both the original essay and the corrected version, with each mistake explained.

Example Feedback:

Student Essay:
"I has a apple."

Corrected Essay:
"I have an apple."

Mistakes and Point Deductions:

"I has" should be "I have" (Verb Conjugation: -1 point)
"a apple" should be "an apple" (Article Use: -0.5 points)
Total Points Deducted: 1.5

Each mistake is explained in detail, with the points deducted listed at the end, so both you and the student can easily see what went wrong and how to improve.

How It Works:

Load your API key for either OpenAI's GPT-4 or GPT-4o.
Select a folder containing student essays and a grading rubric.
Adjust the AI model settings (temperature, top_p).
Grade essays in bulk, with corrections and detailed reports generated automatically.
Requirements:

OpenAI API Key (with GPT-4 or GPT-4o access)
Python 3.x
Tkinter (for the graphical user interface)
Running the Application (PC and Mac)

Step-by-Step Instructions for Beginners:

1. Download and Install Python:

PC (Windows):

Go to Python's website and download Python for Windows.
Run the installer and check the box that says "Add Python to PATH".
Click Install and wait for the installation to complete.
Mac:

Go to Python's website and download Python for macOS.
Open the .pkg installer and follow the instructions.
Open Terminal and type python3 --version to confirm the installation.
2. Download Project Isidore:

On GitHub, click the green Code button on this repository page.
Select Download ZIP.
Extract the downloaded ZIP file to a folder on your computer.
3. Install Required Libraries:

PC (Windows):

Open Command Prompt (search for cmd).
Navigate to the folder where you extracted the project files:
cd path_to_your_folder
(e.g., cd C:\Users\YourName\Downloads\IsidoreGPT)
Install the required libraries:
pip install openai tkinter
Mac:

Open Terminal.
Navigate to the folder where you extracted the project files:
cd path_to_your_folder
(e.g., cd /Users/YourName/Downloads/IsidoreGPT)
Install the required libraries:
pip3 install openai tk
4. Get Your OpenAI API Key:

Sign up or log in to OpenAI.
Navigate to your API keys section and create a new key.
Copy the key and save it in a text file (e.g., apikey.txt).
5. Run the Application:

PC (Windows):

In Command Prompt, navigate to the project folder if you're not already there.
Run the application:
python isidore_grader.py
Mac:

In Terminal, navigate to the project folder if you're not already there.
Run the application:
python3 isidore_grader.py
6. Enter the API Key:

When the application starts, click API Key.
Navigate to the text file where you've saved your API key (e.g., apikey.txt) and select it.
7. Select Student Essays and Grading Rubric:

Click Student Essays to select the folder containing the essays you want to grade.
Click Rubric to select your grading rubric file.
8. Start Grading:

Adjust the model settings if needed (e.g., temperature and top_p).
Click Grade to begin the grading process. The application will generate a detailed report for each essay.
License:

This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

Author: Jonathan Graziola (isidore.gpt@gmail.com)

