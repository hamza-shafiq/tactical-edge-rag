FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /tactical-edge-rag

# Install build tools and other dependencies
RUN apt-get update && \
    apt-get install -y gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install fitz
RUN pip install pymupdf==1.20.0

COPY . .

EXPOSE 8000

# Run the application
CMD ["python3", "app.py", "--host", "0.0.0.0", "--port", "8000"]
