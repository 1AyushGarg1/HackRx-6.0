# main.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
from app import pdfToEmbeddings,answerAQuery

load_dotenv()

API_KEY = os.getenv("API_KEY")

app = FastAPI()

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
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid/missing API key")
    
    try:
        query_engine = pdfToEmbeddings(request_data.documents)
        print(".........................query engine innitialized.......................")
        query_answer = []
        for i in request_data.questions:
            response = answerAQuery(i,query_engine)
            print(response)
            print("........................\n\n.............................")
            query_answer.append(response)
        answers = query_answer
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))