import re
import json
import os



def clean_llm_generated_json(text):
    """
    Given LLM response, extract the json object and 
    Remove JavaScript-style comments (// ...) from a JSON string.
    
    Returns:
        str: Cleaned JSON string without comments.
    """
        
    # Extract JSON using regex
    json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    
    return re.sub(r'//.*', '', json_match)

        
# Read the content from the file which contains the LLMs output
def extract_json_object(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    json_match = clean_llm_generated_json(content)


    if json_match:
        json_text = json_match.group(1)
    
        # Parse the extracted JSON
        extracted_data = json.loads(json_text)
    
        # Print the extracted JSON object
        #print(json.dumps(extracted_data, indent=4))
    else:
        print("No JSON object found in the text.")
        print(file_path)

    return extracted_data

'''
This function
'''
    
def compare_json_files(first_dir, second_dir, filename):
    """
    Reads a JSON file from both directories and checks if they match exactly.
    
    Args:
        first_dir (str): Path to the first directory.
        second_dir (str): Path to the second directory.
        filename (str): Name of the file to compare.

    Returns:
        bool: True if the JSON objects are exactly the same, False otherwise.
    """
    file1_path = os.path.join(first_dir, filename)
    file2_path = os.path.join(second_dir, filename)

    # Check if both files exist
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        #print(file1_path)
        #print(file2_path)
        print(f"One or both files are missing: {filename}")
        print("***************************")
        return False

    try:
        # Read the first file
        with open(file1_path, 'r', encoding='utf-8') as f1:
            json1 = json.load(f1)

        # Read the second file
        json2= extract_json_object(file2_path)

        # Compare JSON objects
        return json1 == json2

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {filename}: {e}")
        return False



def main():

    #GT and llms result dicrectories
    gt_directory = 'data/ground_truth/arxiv/'
    llm_output_directory = 'results/llama3_3_latest' #phi_4_latest'
    
    for filename in os.listdir(gt_directory):
        if filename.endswith(".txt"):    
            result = compare_json_files(gt_directory, llm_output_directory, filename)
            print(f"Files match: {result}")


main()

