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