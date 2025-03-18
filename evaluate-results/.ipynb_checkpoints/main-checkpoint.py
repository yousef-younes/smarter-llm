import re
import json
import os

import comparison_class as cc
import pdb



compare_obj = cc.Compare_two_json_objects()

'''
This function reads groud truth file into a json object
'''
def read_gt_json(gt_dir,filename):

    file = os.path.join(gt_dir,filename)

    if not os.path.exists(file):
        print(f"The file {file} does not exist")

    gotton_json = ""
    try:
        with open(file, 'r', encoding='utf-8') as f1:
            gotton_json = json.load(f1)
    except json.JsonDecodeError as e:
        print(f"Error decoding JSON in {file}")

    return gotton_json

'''
This function cleans the output of the large language models
'''
def clean_llm_output(input_string):
    # Find the start of JSON block i.e. ignore any text that comes before
    start = input_string.find("{")
    
    # Find the end of JSON block i.e., ignore any text that comes after
    end = input_string.rfind("}") + 1
    
    # Extract JSON string
    json_string = input_string[start:end].strip()

    #remove comments from the json string
    json_string = re.sub(r'//.*', '', json_string)

    return json_string

        
'''
This function reads LLM output into a json object
'''
def read_llm_json(llm_dir,filename):
    file = os.path.join(llm_dir,filename)

    if not os.path.exists(file):
        print(f"The file {file} does not exist")

    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    content = clean_llm_output(content)
    gotton_json=""
    try:
        gotton_json = json.loads(content)
    except:
        print("No JSON object found in the text.")
        print(file)

    return gotton_json

'''
This function compares two json objects one from the ground truth and one from the LLM output
'''
def compare_json_objects(gt_directory, llm_output_directory, filename):
    """
    Reads a JSON file from both directories and checks if they match exactly.
    
    Args:
        gt_directory (str): Path to the ground truth directory.
        llm_output_directory (str): Path to the results directory.
        filename (str): Name of the file to compare.

    Returns:
        bool: True if the JSON objects are exactly the same, False otherwise.
    """

    #read ground truth json
    gt_json = read_gt_json(gt_directory, filename)


    #read llm output
    llm_json = read_llm_json(llm_output_directory, filename)

    #compare the two objects
    compare_obj.ground_truth = gt_json
    compare_obj.pred = llm_json
    compare_obj.deep_compare();

def print_comparison_results():
    print(f"Sample counts : {compare_obj.counter}")
    title,name, aff, email = compare_obj.get_avg_scores()
    print("Scores are ordered as Exact Match, BLUE, ROUGE-1, ROUGE-L, F1, Partial Match")
    print(f"The average scores for titles are :{title}")
    print(f"The average scores for the Author names are : {name}")
    print(f"The average scores for the Author affiliations are : {aff}")
    print(f"The average scores for the Author emails are : {email}")
    
    
def main():
    #GT and llms result dicrectories
    gt_directory = '../data/ground_truth/arxiv/'
    llm_output_directory = '../results/on_text_window_5000/gemma2_27b' #phi_4_latest' #mistral_latest' #llama3_3_latest' #phi_4_latest'

    for filename in os.listdir(gt_directory):
        if filename.endswith(".txt"):
            print(filename)
            if filename == 'result_from_first_page_text_d81_arxiv_2307.09288.txt':
                pass #pdb.set_trace()
            compare_json_objects(gt_directory, llm_output_directory, filename)
    #print resutls
    print_comparison_results()
            #print(f"Files match: {result}")

main()
