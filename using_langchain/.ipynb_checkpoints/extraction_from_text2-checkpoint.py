from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

import time


query="""
        You are an expert in **extracting metadata from academic papers**. Your task is to extract the **title** and **author information** from a given research paper.  
        
        For each **author**, extract the following details:
        - **Name**
        - **Affiliation(s)** (An author may have multiple affiliations)
        - **Email address**
        
        ### **Important Constraints:**
        - If any piece of information is **missing or not found**, return **"NA"** instead of leaving it blank.
        - Ensure that affiliations are **correctly mapped** to the corresponding author.
        - The output should be structured in **JSON format** for easy comparison with the ground truth.
        
        ### **Expected Output Format:**
        ```json
        {
          "title": "Extracted Paper Title",
          "authors": [
            {
              "name": "Author Name",
              "affiliations": ["Affiliation 1", "Affiliation 2"],
              "email": "author@example.com"
            },
            {
              "name": "Another Author",
              "affiliations": ["Affiliation A"],
              "email": "NA"
            }
          ]
        }
        ```
        Now, extract and return the structured JSON output based on the given research paper text.
        """


input_folder = "../data/arixv/arxiv_5000" #extracted_texts"
output_folder = "../results/window_5000/mistral_latest"

os.makedirs(output_folder, exist_ok=True)


#this function takes a text file and a model name as input. It uses ollam to run the model on the file's text and return the resutl
def handle_file(input_file,model_name):

    #load text file
    loader=TextLoader(input_file)
    docs = loader.load()

    #split text 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    documents = text_splitter.split_documents(docs)[:5]
    
    embedding_function = OllamaEmbeddings(model=model_name, base_url="http://localhost:3080")
    db = FAISS.from_documents(documents[:5],embedding_function)
            
    llm = Ollama(model=model_name,base_url="http://localhost:3080")
    
    prompt = ChatPromptTemplate.from_template(""" Answer the question based on the provided context only:
                                <context>
                                {context}
                                </context>
                                Question: {input}""")
    
    
    ## create stuff document chain
    document_chain = create_stuff_documents_chain(llm,prompt)
    
    ## create retriever
    retriever = db.as_retriever()
    
    ## create retriever chain
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    
    response= retrieval_chain.invoke({"input":query})

    return response['answer']

    

def main():

    processing_time = 0
    counter = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
    
            input_file = os.path.join(input_folder, filename)
            model_name = "mistral:latest" #"gemma2:27b" #"llama3.3:latest" #"phi4:latest" 	

            start_time = time.time()
            result = handle_file(input_file, model_name)
            end_time = time.time()
            
            elapsed_time = end_time - start_time

            processing_time += elapsed_time
            counter +=1

            
            
            output_path = os.path.join(output_folder, f"result_from_{filename}")
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(result)
            print(f"Results saved to: {output_path}")

    print(f"Average Processing time for a file: {processing_time/counter} seconds")
    
main()
print("completed.")
