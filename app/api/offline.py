from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def offline_home():
    return {
        "mode": "offline",
        "description": "Offline (batch) fraud detection",
        "status": "ready"
    }

@router.get("/dataset-info")
def dataset_info():
    return {
        "samples": 10000,
        "fraud": 800,
        "non_fraud": 9200,
        "mode": "offline"
    }