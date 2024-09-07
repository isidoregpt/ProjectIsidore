Project Isidore
Project Isidore GPT-4 & GPT-4o Grader
Automated Essay Grader Using GPT-4 and GPT-4o with Custom Rubrics

Project Isidore GPT-4 & GPT-4o Grader is a simple Python application that uses OpenAI's GPT-4 or GPT-4o to automatically grade student essays based on customizable grading rubrics. This tool is designed to streamline grading for educators, identifying grammatical errors, improper word usage, and other writing issues, while providing constructive corrections. You can load your own grading rubric and student essays to get detailed feedback and point deductions for specific mistakes.

Key Features:
Custom Rubric Integration: Load your own grading rubric to personalize grading standards.
AI-Powered Essay Correction: Uses either GPT-4 or GPT-4o to identify mistakes and offer clear, concise corrections.
Configurable Model Settings: Adjust model settings like temperature and nucleus sampling (top_p) to refine the grading process.
Automated Grading for Batch Essays: Process multiple student essays at once from a selected folder.
Detailed Feedback: Generates corrected essays with explanations of errors and point deductions.
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
Navigate to the folder where you extracted the project files. You can do this by typing cd path_to_your_folder (e.g., cd C:\Users\YourName\Downloads\IsidoreGPT).
Type the following command and press Enter to install the required libraries:
bash
Copy code
pip install openai tkinter
Mac:

Open Terminal (you can find it using Spotlight Search).
Navigate to the folder where you extracted the project files by typing:
bash

cd path_to_your_folder
(e.g., cd /Users/YourName/Downloads/IsidoreGPT)
Type the following command and press Enter:
bash
Copy code
pip3 install openai tk

4. Get Your OpenAI API Key:
Sign up or log in to OpenAI.
Navigate to your API keys section and create a new key.
Copy the key.
5. Run the Application:
PC (Windows):

In Command Prompt, while still in the folder containing the project files, run the following command:
bash
Copy code
python isidore_grader.py
Mac:

In Terminal, type the following command while in the project folder:
bash
Copy code
python3 isidore_grader.py

6. Enter the API Key:
When the application starts, it will ask you to load your OpenAI API key.
Click API Key and navigate to a text file where you’ve saved your API key (e.g., apikey.txt).

7. Select Student Essays and Grading Rubric:
Click Student Essays to select the folder containing the essays you want to grade.
Click Rubric to select your grading rubric file.

8. Start Grading:
Adjust the model settings if needed (e.g., temperature and top_p).
Click Grade to begin the grading process. The application will generate a detailed report for each essay.

License:
This project is licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
Author: Jonathan Graziola (isidore.gpt@gmail.com)
