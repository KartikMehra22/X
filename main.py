from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Literal
import agent

app = FastAPI(title="SHL Recommender API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic v2 Models
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages must not be empty")
        if len(v) > 8:
            raise ValueError("messages must have at most 8 items")
        if v[-1].role != "user":
            raise ValueError("last message must have role 'user'")
        return v

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Convert Pydantic models to dicts for agent
        messages_dict = [m.model_dump() for m in request.messages]
        
        # Call agent logic
        result = agent.run_agent(messages_dict)
        
        return result
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
