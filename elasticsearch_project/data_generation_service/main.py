from fastapi import FastAPI, HTTPException
from elasticsearch import Elasticsearch
import httpx
from typing import List, Dict
import time
import os
import logging
import asyncio
import requests

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

OLLAMA_API_URL = "http://host.docker.internal:11434/api/generate"


def generate_universe_paragraph(max_retries=3, retry_delay=1) -> Dict:
    prompt = "Generate a cool, interesting paragraph of about 50 words about the universe. Make it engaging and thought-provoking."

    for attempt in range(max_retries):
        try:
            logger.info(f"Generating universe paragraph with prompt: {prompt}")
            response = requests.post(OLLAMA_API_URL, json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            }, timeout=80)

            response.raise_for_status()  # Raises an HTTPError for bad responses

            response_data = response.json()
            content = response_data.get("response", "")
            if not content:
                raise ValueError("Empty response from Ollama")

            logger.info(f"Successfully generated paragraph: {content[:50]}...")
            return {"content": content}
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to generate paragraph after {max_retries} attempts: {str(e)}")

@app.post("/generate-universe-paragraphs")
def generate_universe_paragraphs(count: int = 1) -> List[Dict]:
    global es
    if es is None:
        raise HTTPException(status_code=500, detail="Elasticsearch client not initialized")
    
    paragraphs = []
    for _ in range(count):
        try:
            paragraph = generate_universe_paragraph()
            paragraphs.append(paragraph)
            
            try:
                es.index(index="universe_paragraphs", body=paragraph)
                logger.info(f"Successfully indexed paragraph: {paragraph['content'][:50]}...")
            except Exception as e:
                logger.error(f"Failed to index paragraph: {str(e)}")
                # Consider whether you want to raise an exception here or continue with other paragraphs
        except HTTPException as he:
            logger.error(f"HTTP exception occurred: {str(he)}")
            raise he
    
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