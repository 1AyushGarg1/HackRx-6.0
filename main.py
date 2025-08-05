# main.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pinecone import Pinecone 
import os
from app import build_or_load_index,answer_query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
pinecone_api_key=os.getenv("PINECONE_API_KEY")


API_KEY = os.getenv("API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestModel(BaseModel):
    documents: str
    questions: List[str]

class ResponseModel(BaseModel):
    answers: List[str]

@app.get("/")
def check(request):
    return {
        "response":"Okk" 
    }


@app.post("/hackrx/run", response_model=ResponseModel)
async def run(request_data: RequestModel, authorization: Optional[str] = Header(None)):
    print ("authorization: ", authorization)
    print ("API_KEY: ", f"Bearer {API_KEY}")
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid/missing API key")
    
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "ayush-trail-instance"
        vector_index = await build_or_load_index(request_data.documents, index_name, pc)
        query_engine = vector_index.as_query_engine(similarity_top_k=4)
        query_answer = []
        for i in request_data.questions:
            response = await answer_query(i, query_engine)
            query_answer.append(response)
        return {"answers": query_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    