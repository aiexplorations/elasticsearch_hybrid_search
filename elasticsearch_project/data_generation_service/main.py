from fastapi import FastAPI, HTTPException
from elasticsearch import Elasticsearch
import httpx
from typing import List, Dict
import time
import os
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI()

def create_es_client(max_retries=10, retry_interval=5):
    logger.info("Creating Elasticsearch client")
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

LLAMA_API_URL = "http://host.docker.internal:11434/api/generate"

async def generate_universe_paragraph() -> Dict:
    prompt = """Generate a cool, interesting paragraph of about 50 words about the universe. Make it engaging and thought-provoking."""

    async with httpx.AsyncClient() as client:
        logger.info(f"Generating universe paragraph with prompt: {prompt}")
        logger.info(f"Sending request to {LLAMA_API_URL} with Prompt: {prompt}")
        response = await client.post(LLAMA_API_URL, json={
            "model": "llama3.1:latest",
            "prompt": prompt,
            "stream": False,
        })

    if response.status_code != 200:
        logger.error(f"Failed to generate universe paragraph: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to generate universe paragraph")

    return {"content": response.json()["data"]}

@app.post("/generate-universe-paragraphs")
async def generate_universe_paragraphs(count: int = 1) -> List[Dict]:
    global es
    if es is None:
        raise HTTPException(status_code=500, detail="Elasticsearch client not initialized")
    
    paragraphs = []
    for _ in range(count):
        try:
            logger.info("Generating universe paragraph")
            paragraph = await generate_universe_paragraph()
            paragraphs.append(paragraph)
            
            es.index(index="universe_paragraphs", body=paragraph)
        except Exception as e:
            logger.error(f"Failed to generate or index paragraph: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate or index paragraph: {str(e)}")
    
    return paragraphs

@app.on_event("startup")
async def startup_event():
    global es
    es = create_es_client()
    
    if not es.indices.exists(index="universe_paragraphs"):
        es.indices.create(index="universe_paragraphs", body={
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