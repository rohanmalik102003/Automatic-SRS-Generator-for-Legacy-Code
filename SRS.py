import os
import streamlit as st
import torch
from unsloth import FastLanguageModel

# Configuration
max_seq_length = 3000  # Increased max sequence length
load_in_4bit = True

@st.cache_resource
def load_model():
    """
    Load the FastLanguageModel from a local file.
    """
    st.write("Loading the model...")
    try:
        # Specify the path to your local model directory or file
        model_path = "Llama-3.2-3B-Instruct-bnb-4bit"  # Replace with the actual path

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        model = FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None, None

# Load model and tokenizer
model, tokenizer = load_model()

def preprocess_java_code(java_code):
    """
    Preprocess the Java code to extract meaningful sections.
    Includes class definitions, method signatures, and comments.
    """
    lines = java_code.split("\n")
    extracted_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("package") or stripped_line.startswith("import"):
            continue
        if stripped_line.startswith("public") or stripped_line.startswith("class"):
            extracted_lines.append(stripped_line)
        elif stripped_line.startswith("//") or stripped_line.startswith("/*") or stripped_line.startswith("*"):
            extracted_lines.append(stripped_line)
        elif len(stripped_line) > 0:
            extracted_lines.append(stripped_line)
    return "\n".join(extracted_lines)

def generate_srs(input_code, custom_prompt, max_new_tokens, temperature, top_p):
    """
    Generate an SRS based on the provided Java code.
    """
    model.eval()
    instruction = custom_prompt + f"\n\nCode Analysis:\n{input_code[:600]}..."
    inputs = tokenizer(
        instruction,
        return_tensors="pt",
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    st.title("Automatic Software Documentation for Legacy Code (ISRO)")
    st.write("Upload Java files to generate Software Requirements Specification.")

    uploaded_files = st.file_uploader("Upload Java Files", type=["java"], accept_multiple_files=True)

    custom_user_prompt = st.text_area("Enter Your Custom Prompt (Optional)", "", height=150)

    default_prompt = (
        "Generate a Software Requirements Specification (SRS) compliant with IEEE 830 standards. "
        "The SRS must include the following sections:\n\n"
        "1. Introduction:\n"
        "   - Purpose\n"
        "   - Scope\n"
        "   - Definitions, Acronyms, Abbreviations\n"
        "   - References\n"
        "   - Overview\n"
        "2. Overall Description:\n"
        "   - Product Perspective\n"
        "   - Product Functions\n"
        "   - User Characteristics\n"
        "   - Constraints\n"
        "   - Assumptions and Dependencies\n"
        "3. Specific Requirements:\n"
        "   - Functional Requirements\n"
        "   - Non-functional Requirements\n"
        "   - Interface Requirements\n"
        "4. Appendices\n\n"
        "Base your analysis on the following Java code:\n"
    )

    custom_prompt = custom_user_prompt if custom_user_prompt else default_prompt

    max_new_tokens = st.slider("Max New Tokens", 100, 3000, 512, step=50)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, step=0.1)
    top_p = st.slider("Top-p Sampling", 0.1, 1.0, 0.9, step=0.1)

    if st.button("Generate SRS"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                java_code = uploaded_file.read().decode("utf-8")
                preprocessed_code = preprocess_java_code(java_code)

                st.write(f"Processing file: {uploaded_file.name}...")
                srs_document = generate_srs(preprocessed_code, custom_prompt, max_new_tokens, temperature, top_p)

                if srs_document.strip():
                    st.success(f"SRS generated successfully for {uploaded_file.name}!")
                    st.text_area(f"Generated SRS for {uploaded_file.name}:", value=srs_document, height=300)

                    st.download_button(
                        label=f"Download SRS for {uploaded_file.name}",
                        data=srs_document,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_SRS.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(f"No SRS generated for {uploaded_file.name}. Check the input or model.")
        else:
            st.warning("Please upload at least one Java file.")

if __name__ == "__main__":
    main()
