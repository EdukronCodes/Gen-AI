
# Multi-PDFs 📚 ChatApp AI Agent 🤖

## Abstract
The Multi-PDFs ChatApp AI Agent is a robust web application built using **React** for the frontend and **Django** for the backend, enabling users to interactively chat with multiple PDF documents using advanced AI technologies. By leveraging **Langchain**, **OpenAI models**, and **FAISS Vector DB**, this application enables efficient PDF document processing, text chunking, and conversational queries to generate accurate, context-aware responses in real-time.

The application allows users to upload multiple PDF files, extract text, and manage interactions through an intuitive React-based user interface. The backend, powered by Django, handles text extraction, chunk processing, and communication with AI models for generating responses. The system compares user queries with document content and delivers answers based on semantically similar sections of the PDFs. 

Key features include adaptive chunking for dynamic text management, multi-document conversational queries, and compatibility with both PDF and TXT file formats. With the integration of multiple **OpenAI** models, the app ensures high accuracy and fast performance, transforming static PDFs into interactive, chat-enabled resources for seamless document exploration.


## Project Flow for Customer Support Bot 

1. **User Authentication**: Login or sign-up process using Django authentication.
2. **Upload PDF Documents**: User uploads multiple PDF documents through the React frontend.
3. **PDF Processing**: Backend (Django) extracts text from the uploaded PDF files using PyPDF2.
4. **Text Chunking**: Divide the extracted text into manageable chunks using Langchain.
5. **Embedding Generation**: Generate vector embeddings for each text chunk using OpenAI models.
6. **Store Embeddings**: Store the generated embeddings in FAISS Vector DB for efficient similarity search.
7. **User Query Input**: User inputs a query in the chat interface on the React frontend.
8. **Similarity Search**: Backend retrieves the most relevant text chunks from FAISS DB based on the query.
9. **Response Generation**: Use OpenAI models to generate a response based on the relevant text chunks.
10. **Response Display**: Send the response back to the React frontend and display it in the chat interface.
11. **Continuous Conversation**: Users can continue the conversation, with each query repeating the search and response process.
12. **Session Management**: Track and maintain user chat history and document context for ongoing sessions.
13. **End of Session**: User ends the chat or the session times out, with the option to download chat logs or responses.
