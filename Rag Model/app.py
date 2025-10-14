from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline

#### load pdf

loader = PyPDFLoader(r"C:\Users\bhara\Rag Model\resume.pdf")
docs=loader.load()
print(docs)

#### split text

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(chunks)


####  create embeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
### Store embeddings in vectorstore
vectorstore = FAISS.from_documents(chunks, embeddings)
### Define retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
### load open source llm model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", low_cpu_mem_usage=True)
### Create a text generation pipeline
text_gen_pipeline = pipeline("text-generation", 
                             model=model, 
                             tokenizer=tokenizer, 
                             max_length=2048, 
                             temperature=0.1, top_p=0.95, top_k=50)
### Wrap the pipeline in a HuggingFaceHub LLM
llm = HuggingFaceHub(pipeline=text_gen_pipeline)

### Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                       retriever=retriever, return_source_documents=True)
### Ask a question
while True:
    query = input("Enter your question:")
    if query.lower() == "exit":
        break
    response = qa_chain.run(query)  
    print(response)

# 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
