
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
