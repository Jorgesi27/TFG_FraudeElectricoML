from fastapi import FastAPI
from app.api import online, offline

app = FastAPI(
    title="Electricity Fraud Detection",
    description="Online vs Offline Fraud Detection",
    version="1.0.0"
)

app.include_router(online.router, prefix="/online", tags=["Online Detection"])
app.include_router(offline.router, prefix="/offline", tags=["Offline Detection"])
