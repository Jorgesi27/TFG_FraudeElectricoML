from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def online_home():
    return {
        "mode": "online",
        "description": "Online (streaming) fraud detection",
        "status": "ready"
    }

@router.get("/stream-status")
def stream_status():
    return {
        "current_step": 120,
        "status": "streaming"
    }