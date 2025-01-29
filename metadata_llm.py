import os
from huggingface_hub import InferenceClient

HF_TOKEN = ""

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.2-3B-Instruct"
}

prompt_template = """
The following text is extracted from a research publication and contains information about the title, authors, affiliations, and emails. 
The text might include encoding inconsistencies such as extra spaces, line breaks, and non-standard characters due to PDF conversion.

Please extract the information with the following considerations:

1. **Title**: Identify the title of the publication.
2. **Authors and Affiliations**: Extract the list of authors with their corresponding affiliations. 
    - Authors may have superscript markers (e.g., John Doe^a, Jane Smith^b).
    - Some authors may share affiliations (e.g., a: XYZ University, b: XYZ University).
    - Handle cases where no markers exist and infer affiliations logically.
3. **Email Extraction**: Identify email addresses and associate them with the correct author.

**Example Input:**
Title: Advances in AI Research  
Authors: Dr. John Doe^a, Prof. Jane Smith^a, Dr. Mike Brown^b  
Affiliations: a: XYZ University, b: ABC Institute  
Emails: johndoe@xyz.edu, mikebrown@abc.edu

**Expected Output:**
Title: Advances in AI Research  
Authors:  
- Name: Dr. John Doe, Affiliation: XYZ University, Email: johndoe@xyz.edu  
- Name: Prof. Jane Smith, Affiliation: XYZ University, Email: N/A  
- Name: Dr. Mike Brown, Affiliation: ABC Institute, Email: mikebrown@abc.edu  

---

Text to Process:
{content}

Format the extracted information precisely as per the above structure.
"""

def process_text_file(file_path, model_name):
    """Read and process text file with the selected model using enhanced prompt."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text_content = file.read()


        query = prompt_template.format(content=text_content)


        model_id = MODELS.get(model_name.lower())
        if not model_id:
            raise ValueError(f"Model '{model_name}' not found.")

        client = InferenceClient(model=model_id, token=HF_TOKEN)

        print(f"Processing: {file_path} with model: {model_name}")
        response = client.text_generation(query, max_new_tokens=350, temperature=0.1, top_p=0.9)
        return response.strip()

    except Exception as e:
        print(f"Error processing file {file_path} with model {model_name}: {e}")
        return None


input_folder = "C:/Users/suryanaf/Documents/mist_llm/text"
output_folder = "C:/Users/suryanaf/Documents/mist_llm/output_results"

os.makedirs(output_folder, exist_ok=True)

selected_model = "llama"
#selected_model = "mistral"



for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_folder, filename)
        result = process_text_file(input_path, selected_model)

        if result:
            output_path = os.path.join(output_folder, f"{selected_model}_output_{filename}")
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(result)
            print(f"Results saved to: {output_path}")

print("completeddd.")
