from pydantic import BaseModel


class BaseGeneratePL(BaseModel):
    model_id: str
    text: str
