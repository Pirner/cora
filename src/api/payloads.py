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


class ChatMessage(BaseModel):
    text: str
    sender: Literal['user', 'agent']

class ChatHistory(BaseModel):
    history: List[ChatMessage]

    def convert_to_messages(self) -> List[Message]:
        """
        converts the history into the jinja format for the llm to consume
        :return:
        """
        ret = []
        for c_msg in self.history:
            if c_msg.sender == 'user':
                role = 'user'
            elif c_msg.sender == 'agent':
                role = 'assistant'
            else:
                raise Exception('Unknown user type', c_msg.sender)
            msg = Message(role=role, content=c_msg.text)
            ret.append(msg)
        return ret
