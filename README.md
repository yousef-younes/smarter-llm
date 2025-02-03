# Repository Overview

This repository contains the code for our paper **XXX**, which aims to extract the title and authors' metadata from scientific papers provided in PDF format.

## Directory Structure

- **data/**: Contains PDFs obtained from multiple sources:
  - `arxiv/`: PDFs from Arxiv
  - `ieee/`: PDFs from IEEE
  - `springer/`: PDFs from Springer

- **data-preparation/**: Contains the code used to collect the PDFs and extract text from them.

- **using_langchain/**: Contains the code that leverages the Langchain library to operate different language models on the PDFs and the extracted texts.

- **using_spacy/**: Contains the code used to leverage the Spacy library to perform the task.
