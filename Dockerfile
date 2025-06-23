# RAG_Vetrina/Dockerfile
FROM python:3.12.9-slim

WORKDIR /app

# Installa dipendenze di sistema se necessarie
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY . .

# Crea la directory per il vector store
RUN mkdir -p ./vector_store

# Espone la porta 8000
EXPOSE 8000

# Comando per avviare l'applicazione
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]