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
