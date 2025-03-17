# Automatic-SRS-Generator-for-Legacy-Code
This project automates the generation of Software Requirement Specifications (SRS) for legacy Java code using a Large Language Model (LLM). The tool extracts key components from Java source code and produces structured documentation.

Setup Instructions (Linux)
1️⃣ Clone the Repository
git clone https://github.com/rohanmalik102003/Automatic-SRS-Generator-for-Legacy-Code.git
cd your-repo

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Download the Model (Llama-3.2-3B-Instruct)
git clone https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit
📌 Ensure the model is inside models/ before running the application.

Usage
Start the Streamlit Application
streamlit run SRS.py

🔹 Open the browser at http://localhost:8501
🔹 Upload Java files to generate an SRS document

