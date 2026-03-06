from pydantic import BaseModel

class StatusResponse(BaseModel):
    mode: str
    status: str
