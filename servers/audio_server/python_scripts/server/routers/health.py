from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Whisper CTranslate2 API ready"}