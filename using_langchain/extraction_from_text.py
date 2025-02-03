from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os


query="""
        Extract the title and author information of this paper and produce the result in the following format
        
        **Expected Output:**
        Title: Advances in AI Research  
        Authors:  
        - Name: Dr. John Doe, Affiliation: XYZ University, Email: johndoe@xyz.edu  
        - Name: Prof. Jane Smith, Affiliation: XYZ University, Email: N/A  
        - Name: Dr. Mike Brown, Affiliation: ABC Institute, Email: mikebrown@abc.edu  
        
        if some info is not available write NA
        """


input_folder = "../data/extracted_texts"
output_folder = "../results/llama3_3_latest" #"mistral_latest" #"../results/llama3_3_latest" 

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
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
    
            input_file = os.path.join(input_folder, filename)
            model_name =  "llama3.3:latest" #"mistral:latest" phi4:latest	#"gemma2:27b"	
    
            result = handle_file(input_file, model_name)
            
            output_path = os.path.join(output_folder, f"result_from_{filename}")
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(result)
            print(f"Results saved to: {output_path}")

main()
print("completed.")
