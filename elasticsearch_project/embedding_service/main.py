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
