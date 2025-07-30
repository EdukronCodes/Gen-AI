# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    unzip \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for some dependencies
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install spaCy English model
RUN python -m spacy download en_core_web_sm

# Install additional system packages for specific projects
RUN apt-get update && apt-get install -y \
    # For OpenCV
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # For RDKit
    libboost-all-dev \
    libeigen3-dev \
    # For Selenium
    chromium \
    chromium-driver \
    # For monitoring
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/chroma_db \
    /app/toxicity_chroma_db \
    /app/resume_chroma_db \
    /app/patient_chroma_db \
    /app/surgical_chroma_db \
    /app/customer_support_chroma_db \
    /app/fraud_chroma_db \
    /app/retail_chroma_db \
    /app/medical_knowledge \
    /app/data/toxicity \
    /app/data/job_market \
    /app/data/patients \
    /app/data/surgical \
    /app/data/products \
    /app/data/insurance \
    /app/data/retail \
    /app/uploads \
    /app/logs \
    /app/models \
    /app/results

# Copy application code
COPY . .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash ml_user \
    && chown -R ml_user:ml_user /app

# Switch to non-root user
USER ml_user

# Expose ports
EXPOSE 8000 8001 8002 8003 8004 8005 8006 8007

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "agentic_medical_chatbot/rag_medical_chatbot.py"] 