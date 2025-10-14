### DOCUMENT PREPROCESSOR UTILITIES FOR RAG SYSTEM

import os
import re
import pyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
#from utils.logger import setup_logger


class DocumentPreprocessor:
    def __init__(self)
      self.supported_extensions=['.pdf', '.txt', '.csv', '.xlsx', '.docx']
      self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                        chunk_overlap=200)
      
    def extract_text_from_files(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.pdf':
            return self.extract_pdf(file_path)
        elif file_extension == '.txt':
            return self.extract_text(file_path)
        elif file_extension == '.csv':
            return self.extract_csv(file_path)
        elif file_extension == '.xlsx':
            return self.extract_excel(file_path)
        elif file_extension == '.docx':
            return self.extract_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def extract_pdf(self, file_path):  
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    def extract_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [text]
    def extract_csv(self, file_path):
        import pandas as pd
        df = pd.read_csv(file_path)
        text = df.to_string()
        return [text]
    def extract_excel(self, file_path):
        import pandas as pd
        df = pd.read_excel(file_path)
        text = df.to_string()
        return [text]
    def extract_docx(self, file_path):
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        return documents
    def chunk_documents(self, documents):
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_uploaded_file(self, file_path):
        documents = self.extract_text_from_files(file_path)
        chunks = self.chunk_documents(documents)
        return chunks   
    def validate_file(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError("File is empty")
        return True
    