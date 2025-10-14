FROM python:3.11-slim
WORKDIR /app
ENV streamlit_server_port=8501
    streamlit_server_address=0.0.0.0

COPY . .


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN MKDIR -P LOGS UPLAODS  TEMP DATA CHROM_DB

EXPOSE 8501

### RUN THE APPLICATION

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=

