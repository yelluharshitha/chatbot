"""
main.py - FastAPI Backend using memory.py
Install: pip install fastapi uvicorn langchain langchain-anthropic python-dotenv
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import uuid
import os
from memory import ChatMemory

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store ChatMemory instances per session
sessions: Dict[str, ChatMemory] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    memory_size: int

class ClearMemoryRequest(BaseModel):
    session_id: str

class HistoryResponse(BaseModel):
    session_id: str
    history: str
    message_count: int


@app.get("/")
async def root():
    return {
        "message": "LangChain Memory API Running",
        "endpoints": {
            "/chat": "POST - Send message",
            "/clear": "POST - Clear memory",
            "/history/{session_id}": "GET - Get chat history"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get AI response with memory
    """
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create new ChatMemory if session doesn't exist
        if session_id not in sessions:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise HTTPException(
                    status_code=500, 
                    detail="ANTHROPIC_API_KEY not found in environment"
                )
            sessions[session_id] = ChatMemory(api_key=api_key)
        
        # Get chat instance
        chat = sessions[session_id]
        
        # Get response using LangChain memory
        response = chat.chat(request.message)
        
        # Get memory size
        memory_size = chat.get_memory_size()
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            memory_size=memory_size
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_memory(request: ClearMemoryRequest):
    """
    Clear conversation memory for a session
    """
    try:
        session_id = request.session_id
        
        if session_id in sessions:
            # Clear memory using LangChain method
            sessions[session_id].clear_memory()
            return {
                "message": "Memory cleared successfully",
                "session_id": session_id
            }
        else:
            return {
                "message": "Session not found",
                "session_id": session_id
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """
    Get full conversation history for a session
    """
    try:
        if session_id not in sessions:
            raise HTTPException(
                status_code=404, 
                detail="Session not found"
            )
        
        chat = sessions[session_id]
        
        # Get history from LangChain memory
        history = chat.get_chat_history()
        message_count = chat.get_memory_size()
        
        return HistoryResponse(
            session_id=session_id,
            history=str(history),
            message_count=message_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete entire session
    """
    if session_id in sessions:
        del sessions[session_id]
        return {
            "message": "Session deleted successfully",
            "session_id": session_id
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions")
async def list_sessions():
    """
    List all active sessions
    """
    session_info = []
    for session_id, chat in sessions.items():
        session_info.append({
            "session_id": session_id,
            "message_count": chat.get_memory_size()
        })
    
    return {
        "total_sessions": len(sessions),
        "sessions": session_info
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
