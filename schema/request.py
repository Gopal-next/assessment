from pydantic import BaseModel

# request of questions
class Message(BaseModel):
    role:str
    content:str

class ChatRequest(BaseModel):
    messages:list[Message]

# response of questions
