import re
import json

# Read the content from the file which contains the LLMs output
def extract_json_object(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Extract JSON using regex
    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
    
    if json_match:
        json_text = json_match.group(1)
    
        # Parse the extracted JSON
        extracted_data = json.loads(json_text)
    
        # Print the extracted JSON object
        print(json.dumps(extracted_data, indent=4))
    else:
        print("No JSON object found in the text.")



def read_json_object_from_GT(file_path):
    
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()  # Read entire file as a string
    
    # Parse the string as JSON
    data = json.loads(text_content)
    
    # Now 'data' is a Python dictionary
    print(data['title'])  # Access specific fields


file = 'data/ground_truth/arxiv/result_from_first_page_text_d11_arxiv_1804.00015.txt'
read_json_object_from_GT(file)

#GT folder
gt_folder = 'data/ground_truth/arxiv/'
llm_output_folder = 'smartER/smarter-llm/results/window_5000/phi_4_latest'

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):

        input_file = os.path.join(input_folder, filename)
