
# Multi-PDFs 📚 ChatApp AI Agent 🤖

## Abstract
The Multi-PDFs ChatApp AI Agent is a robust web application built using **React** for the frontend and **Django** for the backend, enabling users to interactively chat with multiple PDF documents using advanced AI technologies. By leveraging **Langchain**, **OpenAI models**, and **FAISS Vector DB**, this application enables efficient PDF document processing, text chunking, and conversational queries to generate accurate, context-aware responses in real-time.

The application allows users to upload multiple PDF files, extract text, and manage interactions through an intuitive React-based user interface. The backend, powered by Django, handles text extraction, chunk processing, and communication with AI models for generating responses. The system compares user queries with document content and delivers answers based on semantically similar sections of the PDFs. 

Key features include adaptive chunking for dynamic text management, multi-document conversational queries, and compatibility with both PDF and TXT file formats. With the integration of multiple **OpenAI** models, the app ensures high accuracy and fast performance, transforming static PDFs into interactive, chat-enabled resources for seamless document exploration.


## Project Flow for Customer Support Bot 

1. **User Authentication**: Login or sign-up process using Django authentication.

To implement a user authentication system using Django, you can follow these steps. This guide will help you set up a login and sign-up process using Django's built-in authentication system.

### Step 1: Set Up Your Django Project

1. **Create a Django Project:**
   ```bash
   django-admin startproject myproject
   ```

2. **Create a Django App:**
   ```bash
   cd myproject
   python manage.py startapp accounts
   ```

3. **Add the App to Installed Apps:**
   In `settings.py`, add `'accounts'` to the `INSTALLED_APPS` list.

### Step 2: Set Up User Authentication

1. **Create User Model (Optional):**
   If you need a custom user model, define it in `models.py` of the `accounts` app. Otherwise, you can use Django's default user model.

2. **Create Forms for Login and Sign-Up:**
   In `forms.py` of the `accounts` app, create forms for user registration and login using Django's `UserCreationForm` and `AuthenticationForm`.

   ```python
   from django import forms
   from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
   from django.contrib.auth.models import User

   class SignUpForm(UserCreationForm):
       class Meta:
           model = User
           fields = ('username', 'password1', 'password2')

   class LoginForm(AuthenticationForm):
       pass
   ```

### Step 3: Create Views for Authentication

1. **Create Views for Sign-Up and Login:**
   In `views.py` of the `accounts` app, create views to handle user registration and login.

   ```python
   from django.shortcuts import render, redirect
   from django.contrib.auth import login, authenticate
   from .forms import SignUpForm, LoginForm

   def signup_view(request):
       if request.method == 'POST':
           form = SignUpForm(request.POST)
           if form.is_valid():
               form.save()
               username = form.cleaned_data.get('username')
               password = form.cleaned_data.get('password1')
               user = authenticate(username=username, password=password)
               login(request, user)
               return redirect('home')
       else:
           form = SignUpForm()
       return render(request, 'accounts/signup.html', {'form': form})

   def login_view(request):
       if request.method == 'POST':
           form = LoginForm(data=request.POST)
           if form.is_valid():
               user = form.get_user()
               login(request, user)
               return redirect('home')
       else:
           form = LoginForm()
       return render(request, 'accounts/login.html', {'form': form})
   ```

### Step 4: Create Templates

1. **Create HTML Templates:**
   Create `signup.html` and `login.html` in the `templates/accounts` directory.

   ```html
   <!-- signup.html -->
   <form method="post">
       {% csrf_token %}
       {{ form.as_p }}
       <button type="submit">Sign Up</button>
   </form>

   <!-- login.html -->
   <form method="post">
       {% csrf_token %}
       {{ form.as_p }}
       <button type="submit">Log In</button>
   </form>
   ```

### Step 5: Configure URLs

1. **Set Up URLs:**
   In `urls.py` of the `accounts` app, define the URL patterns for login and sign-up views.

   ```python
   from django.urls import path
   from . import views

   urlpatterns = [
       path('signup/', views.signup_view, name='signup'),
       path('login/', views.login_view, name='login'),
   ]
   ```

2. **Include App URLs in Project URLs:**
   In the project's `urls.py`, include the `accounts` app URLs.

   ```python
   from django.contrib import admin
   from django.urls import path, include

   urlpatterns = [
       path('admin/', admin.site.urls),
       path('accounts/', include('accounts.urls')),
   ]
   ```

### Step 6: Test the Authentication System

1. **Run the Server:**
   ```bash
   python manage.py runserver
   ```

2. **Access the Authentication Pages:**
   Visit `/accounts/signup/` to sign up and `/accounts/login/` to log in.

This setup provides a basic user authentication system using Django's built-in features. You can further customize the forms, views, and templates to suit your application's needs.

2. **Upload PDF Documents**: User uploads multiple PDF documents through the React frontend.

To implement a feature where users can upload multiple PDF documents through a React frontend, you can follow these steps:

### Step 1: Set Up the React Frontend

1. **Create a New React App:**
   Use Create React App to set up a new React project.
   ```bash
   npx create-react-app pdf-upload
   cd pdf-upload
   ```

2. **Install Axios:**
   Axios is a promise-based HTTP client for the browser and Node.js, which will be used to send the PDF files to the backend.
   ```bash
   npm install axios
   ```

3. **Create a File Upload Component:**
   Create a new component, `FileUpload.js`, to handle the file upload functionality.

   ```jsx
   import React, { useState } from 'react';
   import axios from 'axios';

   const FileUpload = () => {
     const [selectedFiles, setSelectedFiles] = useState([]);

     const handleFileChange = (event) => {
       setSelectedFiles(event.target.files);
     };

     const handleUpload = async () => {
       const formData = new FormData();
       for (let i = 0; i < selectedFiles.length; i++) {
         formData.append('files', selectedFiles[i]);
       }

       try {
         const response = await axios.post('/upload', formData, {
           headers: {
             'Content-Type': 'multipart/form-data',
           },
         });
         console.log('Upload successful:', response.data);
       } catch (error) {
         console.error('Error uploading files:', error);
       }
     };

     return (
       <div>
         <input type="file" multiple onChange={handleFileChange} />
         <button onClick={handleUpload}>Upload</button>
       </div>
     );
   };

   export default FileUpload;
   ```

4. **Use the File Upload Component:**
   Include the `FileUpload` component in your main application file, `App.js`.

   ```jsx
   import React from 'react';
   import FileUpload from './FileUpload';

   function App() {
     return (
       <div className="App">
         <h1>Upload PDF Documents</h1>
         <FileUpload />
       </div>
     );
   }

   export default App;
   ```

### Step 2: Set Up the Backend

1. **Create a Backend Server:**
   You can use Node.js with Express to create a simple backend server to handle file uploads.

2. **Install Required Packages:**
   Install Express and Multer (a middleware for handling `multipart/form-data`, which is used for uploading files).

   ```bash
   npm install express multer
   ```

3. **Create the Server:**
   Create a file, `server.js`, to set up the server and handle file uploads.

   ```javascript
   const express = require('express');
   const multer = require('multer');
   const path = require('path');

   const app = express();
   const upload = multer({ dest: 'uploads/' });

   app.post('/upload', upload.array('files'), (req, res) => {
     try {
       res.status(200).json({ message: 'Files uploaded successfully' });
     } catch (error) {
       res.status(500).json({ error: 'Error uploading files' });
     }
   });

   const PORT = process.env.PORT || 5000;
   app.listen(PORT, () => {
     console.log(`Server running on port ${PORT}`);
   });
   ```

4. **Run the Backend Server:**
   Start the server by running the following command:
   ```bash
   node server.js
   ```

### Step 3: Connect Frontend and Backend

1. **Proxy Setup:**
   If your React app and backend server are running on different ports, you may need to set up a proxy in the `package.json` of your React app to avoid CORS issues.

   ```json
   "proxy": "http://localhost:5000"
   ```

2. **Test the Upload:**
   Run your React app and backend server, and test the file upload functionality by selecting multiple PDF files and clicking the upload button.

This setup allows users to upload multiple PDF documents through a React frontend, with the files being sent to a backend server for processing or storage.





3. **PDF Processing**: Backend (Django) extracts text from the uploaded PDF files using PyPDF2.

To implement PDF processing in a Django backend using PyPDF2 to extract text from uploaded PDF files, follow these steps:

### Step 1: Set Up Your Django Project

1. **Create a Django Project and App:**
   If you haven't already, create a Django project and an app to handle the PDF processing.

   ```bash
   django-admin startproject myproject
   cd myproject
   python manage.py startapp pdf_processor
   ```

2. **Add the App to Installed Apps:**
   In `settings.py`, add `'pdf_processor'` to the `INSTALLED_APPS` list.

### Step 2: Install PyPDF2

1. **Install PyPDF2:**
   Use pip to install the PyPDF2 library, which will be used to extract text from PDF files.

   ```bash
   pip install PyPDF2
   ```

### Step 3: Create a View to Handle PDF Upload and Processing

1. **Create a View:**
   In `views.py` of the `pdf_processor` app, create a view to handle the file upload and text extraction.

   ```python
   from django.shortcuts import render
   from django.http import JsonResponse
   from PyPDF2 import PdfReader
   import os

   def upload_pdf(request):
       if request.method == 'POST' and request.FILES['pdf']:
           pdf_file = request.FILES['pdf']
           reader = PdfReader(pdf_file)
           text = ''
           for page in reader.pages:
               text += page.extract_text()
           return JsonResponse({'text': text})
       return render(request, 'pdf_processor/upload.html')
   ```

### Step 4: Create a Template for PDF Upload

1. **Create an HTML Template:**
   Create `upload.html` in the `templates/pdf_processor` directory.

   ```html
   <form method="post" enctype="multipart/form-data">
       {% csrf_token %}
       <input type="file" name="pdf" accept="application/pdf">
       <button type="submit">Upload PDF</button>
   </form>
   ```

### Step 5: Configure URLs

1. **Set Up URLs:**
   In `urls.py` of the `pdf_processor` app, define the URL pattern for the upload view.

   ```python
   from django.urls import path
   from . import views

   urlpatterns = [
       path('upload/', views.upload_pdf, name='upload_pdf'),
   ]
   ```

2. **Include App URLs in Project URLs:**
   In the project's `urls.py`, include the `pdf_processor` app URLs.

   ```python
   from django.contrib import admin
   from django.urls import path, include

   urlpatterns = [
       path('admin/', admin.site.urls),
       path('pdf_processor/', include('pdf_processor.urls')),
   ]
   ```

### Step 6: Test the PDF Processing System

1. **Run the Server:**
   Use `python manage.py runserver` to start the Django development server.

2. **Access the PDF Upload Page:**
   Visit `/pdf_processor/upload/` to upload a PDF and extract its text.

This setup allows you to upload PDF files through a Django frontend and extract text from them using PyPDF2 in the backend. You can further customize the views and templates to suit your application's needs.


4. **Text Chunking**: Divide the extracted text into manageable chunks using Langchain.

from langchain.text_splitter import CharacterTextSplitter

# Sample text extracted from a PDF
extracted_text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

# Initialize the text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

# Split the text into chunks
chunks = text_splitter.split_text(extracted_text)
chunks





5. **Embedding Generation**: Generate vector embeddings for each text chunk using OpenAI models.

# Update the code to use the new OpenAI API interface

def generate_embeddings_v1(chunks):
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

# Generate embeddings for the text chunks using the updated function
embeddings_v1 = generate_embeddings_v1(chunks)
embeddings_v1[:2]  # Display the first two embeddings for brevity



6. **Store Embeddings**: Store the generated embeddings in FAISS Vector DB for efficient similarity search.
7. **User Query Input**: User inputs a query in the chat interface on the React frontend.
8. **Similarity Search**: Backend retrieves the most relevant text chunks from FAISS DB based on the query.
9. **Response Generation**: Use OpenAI models to generate a response based on the relevant text chunks.
10. **Response Display**: Send the response back to the React frontend and display it in the chat interface.
11. **Continuous Conversation**: Users can continue the conversation, with each query repeating the search and response process.
12. **Session Management**: Track and maintain user chat history and document context for ongoing sessions.
13. **End of Session**: User ends the chat or the session times out, with the option to download chat logs or responses.



