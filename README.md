
# Project Isidore GPT-4 & GPT-4o Grader

**Automated Essay Grader Using GPT-4 and GPT-4o with Custom Rubrics**

Project Isidore GPT-4 & GPT-4o Grader is a Python application that uses OpenAI's GPT-4 or GPT-4o to automatically grade student essays based on customizable grading rubrics. This tool streamlines grading for educators, identifying grammatical errors, improper word usage, and other writing issues, while providing constructive corrections. You can load your own grading rubric and student essays to get detailed feedback and point deductions for specific mistakes.

---

## Acknowledgment

This project was written with the assistance of **ChatGPT-4** and **ChatGPT-4o**, developed by OpenAI. These AI tools helped generate code, refine features, and structure documentation to make the project more accessible and efficient.

While AI was a significant part of the development process, all decisions, design choices, and customizations were made by the author, Jonathan Graziola.

---

## Disclaimer: Consent, Anonymization, and Compliance with Legal Frameworks

**Important:** This project is designed to assist with automated grading of essays using GPT-4 and GPT-4o. When using this tool, it is crucial to ensure compliance with applicable privacy laws and regulations, such as the **Family Educational Rights and Privacy Act (FERPA)** in the United States.

### Key Considerations:
1. **Consent**: Ensure you have explicit consent from students, parents, or guardians (where applicable) before processing essays that contain personally identifiable information (PII). This tool is not designed to handle sensitive student data without proper consent.
   
2. **Anonymization**: To protect privacy, it is recommended that any essays processed through this tool be anonymized. Remove or mask PII such as names, student ID numbers, and any other identifying information before submission.

3. **Compliance with FERPA and Other Privacy Laws**: If you are using this tool in an educational setting in the United States, ensure you are compliant with **FERPA**, which protects the privacy of student education records. For other countries, comply with relevant data protection laws (e.g., **GDPR** in Europe).
   
4. **Handling PII**: This tool does not store or transmit any data beyond the grading session, but users are responsible for ensuring the proper handling and protection of PII when using OpenAI's services. Ensure you understand OpenAI's privacy policy and how it applies to the data you provide.

### Examples of PII:
- Full names
- Student ID numbers
- Birth dates
- Addresses or contact details

**Best Practice**: Before using this tool, ensure that any identifiable information is removed from the essays. For example, replace full names with "Student A" or other pseudonyms, and avoid submitting any sensitive information through the system.

### Limitation of Liability:
The author of this tool, Jonathan Graziola, and the contributors are not responsible for the misuse of this application or any violations of privacy laws that may arise from improper use. Users of this tool are solely responsible for ensuring compliance with applicable laws and for obtaining all necessary consents.

---

# Key Features:

### 1. Custom Rubric Integration:
Load your own grading rubric to personalize grading standards. The rubric is a simple text file that outlines the grading rules you want the system to follow. For example, you can specify how many points to deduct for incorrect grammar, spelling, or punctuation.

**Example Rubric:**
```plaintext
Grammar: -1 point for each error
Spelling: -0.5 points for each mistake
Punctuation: -0.5 points for each mistake
Verb tense: -1 point for each incorrect use
Wrong word: -1 point for each instance
Incomplete sentences: -2 points
