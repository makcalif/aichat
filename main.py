from fastapi import FastAPI
from pydantic import BaseModel
from rag import RAGPipeline
from dotenv import load_dotenv
from mangum import Mangum

load_dotenv()
app = FastAPI() 

docs = [
    "FastAPI is a modern Python web framework.",
    "FAISS helps with similarity search.",
    "AFUC is an urgent care in Coppel, Texas: it is a walk-in clinic that provides immediate care for non-life-threatening conditions."
]

pipeline = RAGPipeline(docs)
print(pipeline.query("What is FastAPI?"))

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    # Your logic here, e.g. call RAG pipeline
    answer = pipeline.query(query.question)
    return {"answer": answer}

lambda_handler = Mangum(app)  # For AWS Lambda deployment