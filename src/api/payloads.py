from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class BaseGeneratePL(BaseModel):
    model_id: str
    text: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(..., description="JSON Schema-like parameters object")


class Tool(BaseModel):
    type: Literal["function"]
    function: FunctionDefinition


class MessagePL(BaseModel):
    model_id: str
    messages: List[Message]
    tools: Optional[List[Tool]] = None


class StructuredOutputPL(BaseModel):
    model_id: str
    messages: List[Message]
    json_schema: str
