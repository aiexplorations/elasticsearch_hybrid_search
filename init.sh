#!/bin/bash

# Create project directory
mkdir -p elasticsearch_project
cd elasticsearch_project

# Create docker-compose.yml
cat << EOF > docker-compose.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - esdata:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9200"]
      interval: 30s
      timeout: 10s
      retries: 5

  data_generation_service:
    build: ./data_generation_service
    ports:
      - "8000:8000"
    depends_on:
      elasticsearch:
        condition: service_healthy
    environment:
      - ELASTICSEARCH_URL=http://elasticsearch:9200
    extra_hosts:
      - "host.docker.internal:host-gateway"

  embedding_service:
    build: ./embedding_service
    ports:
      - "8001:8001"

  backend_service:
    build: ./backend_service
    ports:
      - "8002:8002"
    depends_on:
      elasticsearch:
        condition: service_healthy
      embedding_service:
        condition: service_started
    environment:
      - ELASTICSEARCH_URL=http://elasticsearch:9200

volumes:
  esdata:
    driver: local
EOF

# Create data_generation_service
mkdir -p data_generation_service
cat << EOF > data_generation_service/Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat << EOF > data_generation_service/requirements.txt
fastapi
uvicorn
elasticsearch==8.8.0
httpx
EOF


# Update the data_generation_service/main.py file
cat << EOF > data_generation_service/main.py
from fastapi import FastAPI, HTTPException
from elasticsearch import Elasticsearch
import httpx
from typing import List, Dict
import time
import os
import random

app = FastAPI()

def create_es_client(max_retries=10, retry_interval=5):
    es_url = os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
    for attempt in range(max_retries):
        try:
            es = Elasticsearch([es_url])
            es.info()
            print(f"Successfully connected to Elasticsearch on attempt {attempt + 1}")
            return es
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
    raise Exception("Failed to connect to Elasticsearch after multiple attempts")

es = None

LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://host.docker.internal:11434/api/generate")

def generate_fallback_resume() -> str:
    names = ["John Doe", "Jane Smith", "Alice Johnson", "Bob Brown"]
    jobs = ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Specialist"]
    skills = ["Python", "Java", "JavaScript", "SQL", "Machine Learning", "Data Analysis", "Project Management"]
    
    name = random.choice(names)
    job = random.choice(jobs)
    skill_list = random.sample(skills, 3)
    
    resume = f"""
    Name: {name}
    Current Position: {job}
    
    Summary:
    Experienced professional with a strong background in {job.lower()}. Skilled in {', '.join(skill_list)}.
    
    Work Experience:
    - {job} at Tech Corp (2018-present)
    - Junior {job} at Startup Inc. (2015-2018)
    
    Education:
    Bachelor's degree in Computer Science, University of Technology (2015)
    
    Skills:
    {', '.join(skill_list)}
    """
    return resume

async def generate_resume() -> Dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(LLAMA_API_URL, json={
                "model": "llama3.1:latest",
                "prompt": "Generate a resume with the following sections: Personal Information, Summary, Work Experience, Education, Skills. Make it realistic and varied. Format it as plain text.",
                "max_tokens": 500,
                "temperature": 0.7
            })
            response.raise_for_status()
            return {"content": response.json()["data"]}
    except Exception as e:
        print(f"Failed to generate resume using LLaMA API: {str(e)}")
        print("Using fallback resume generator.")
        return {"content": generate_fallback_resume()}

@app.post("/generate-resumes")
async def generate_resumes(count: int = 1) -> List[Dict]:
    global es
    if es is None:
        raise HTTPException(status_code=500, detail="Elasticsearch client not initialized")
    
    resumes = []
    for _ in range(count):
        try:
            resume = await generate_resume()
            resumes.append(resume)
            
            es.index(index="resumes", body=resume)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate or index resume: {str(e)}")
    
    return resumes

@app.on_event("startup")
async def startup_event():
    global es
    es = create_es_client()
    
    if not es.indices.exists(index="resumes"):
        es.indices.create(index="resumes", body={
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "vector": {
                        "type": "dense_vector",
                        "dims": 384
                    }
                }
            }
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Create embedding_service
mkdir -p embedding_service
cat << EOF > embedding_service/Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
EOF

cat << EOF > embedding_service/requirements.txt
fastapi
uvicorn
sentence_transformers
torch
EOF

cat << EOF > embedding_service/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()
model = SentenceTransformer('intfloat/e5-large-v2')

class TextInput(BaseModel):
    text: str

@app.post("/compute-embedding")
async def compute_embedding(input: TextInput):
    with torch.no_grad():
        embedding = model.encode(input.text)
    return {"embedding": embedding.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
EOF

# Create backend_service
mkdir -p backend_service
cat << EOF > backend_service/Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
EOF

cat << EOF > backend_service/requirements.txt
fastapi
uvicorn
elasticsearch==8.8.0
httpx
EOF


# Update the backend_service/main.py file
cat << EOF > backend_service/main.py
from fastapi import FastAPI, HTTPException
from elasticsearch import Elasticsearch
import httpx
from typing import List, Dict
import os

app = FastAPI()
es = Elasticsearch([os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")])
embedding_service_url = "http://embedding_service:8001/compute-embedding"

async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(embedding_service_url, json={"text": text})
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to compute embedding")
    return response.json()["embedding"]

@app.get("/search")
async def search(query: str):
    query_embedding = await get_embedding(query)

    search_body = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"content": query}},
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                                "params": {"query_vector": query_embedding}
                            }
                        }
                    }
                ]
            }
        }
    }

    try:
        search_results = es.search(index="resumes", body=search_body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    formatted_results = []
    for hit in search_results["hits"]["hits"]:
        formatted_results.append({
            "content": hit["_source"]["content"],
            "score": hit["_score"]
        })

    return {"results": formatted_results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
EOF

echo "Project structure and files have been created successfully!"
EOF

chmod +x init.sh