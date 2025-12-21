import os
from pyPdf2 import PdfReader
from langchain.document_loaders import PyPDFLoader

class CustomPDFLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def read_pdf(self,file_path):   
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def read_all_pdfs(self):
        all_texts = {}
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(self.folder_path, filename)
                text = self.read_pdf(file_path)
                all_texts[filename] = text
        return all_texts
if __name__ == "__main__":
    folder_path = "C:/Users/ADMIN/Desktop/Auguust 2025 Gen AI/genai_project/data/raw"
    loader = CustomPDFLoader(folder_path)
    pdf_texts = loader.read_all_pdfs()
    for filename, text in pdf_texts.items():
        print(f"Contents of {filename}:\n{text[:500]}...\n")

### validation checkpoint 
### return a table which has all the pdf files and original count of the words vs extrcted words
###  cosine score or eucliedean distance score between original vs extracted



### pypdf2
#### pdfplumber
#### pdfminer 
### fitz (pymupdf)
### reportlab
### camlelot-py
#### pdf2table