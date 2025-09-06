from fastapi import APIRouter, UploadFile, Form
from app.services.rag_service import process_document, answer_question
from pydantic import BaseModel

router = APIRouter()

class AskRequest(BaseModel):
    doc_id: str
    question: str

@router.post("/upload")
async def upload_document(file: UploadFile):
    doc_id,summary = process_document(file.file)
    return {"message": "Uploaded successfully", "doc_id": doc_id,"summary":summary}


@router.post("/ask")
async def ask_question(req: AskRequest):
    answer = answer_question(req.doc_id, req.question)
    return {"answer": answer}
