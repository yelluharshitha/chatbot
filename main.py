"""
main.py - FastAPI Server with History Tracking
"""

import logging
import sys
import os
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from agent import ChatbotAgent

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("chatbot.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    logger.error("NVIDIA_API_KEY not found in .env file!")
    sys.exit(1)

# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"

class ChatResponse(BaseModel):
    success: bool
    response: str
    tool_used: str
    message_count: int
    thread_id: str
    timestamp: str

class DetailedHistoryResponse(BaseModel):
    thread_id: str
    session_id: str
    total_messages: int
    conversation: List[dict]
    created_at: str
    last_active: str

# FastAPI App
app = FastAPI(
    title="AI Chatbot with History Tracking",
    description="Tool-calling chatbot that remembers conversations",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = None

@app.on_event("startup")
async def startup_event():
    global chatbot
    logger.info(" Starting server...")
    try:
        chatbot = ChatbotAgent(NVIDIA_API_KEY)
        logger.info("Server ready!")
        print("\n" + "="*60)
        print("SERVER IS RUNNING!")
        print("="*60)
        print(" URL: http://localhost:8000")
        print(" Docs: http://localhost:8000/docs")
        print("="*60 + "\n")
    except Exception as e:
        logger.error(f" Failed: {e}")
        raise

# API Endpoints

@app.get("/", tags=["General"])
def home():
    """API Home"""
    return {
        "message": "AI Chatbot API with Complete History Tracking",
        "version": "2.0.0",
        "status": "online",
        "features": {
            "history_tracking": {
                "user_query": "✓",
                "bot_response": "✓",
                "tool_used": "✓",
                "session_id": "✓",
                "timestamp": "✓"
            }
        },
        "docs": "http://localhost:8000/docs"
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health Check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(chatbot.list_sessions()) if chatbot else 0
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest):
    """
    Send message to chatbot
    
    Tracks:
    - User query
    - Bot response
    - Tool used
    - Session ID
    - Timestamp
    """
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    logger.info(f" Message from {request.thread_id}")
    
    try:
        result = chatbot.chat(request.message, request.thread_id)
        
        return ChatResponse(
            success=result["success"],
            response=result["response"],
            tool_used=result["tool_used"],
            message_count=result["message_count"],
            thread_id=result["thread_id"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f" Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{thread_id}", tags=["History"])
def get_history(thread_id: str):
    """Get simple text history"""
    if not chatbot:
        raise HTTPException(status_code=503)
    
    history = chatbot.get_history(thread_id)
    return {
        "thread_id": thread_id,
        "history": history
    }


@app.get("/history/{thread_id}/detailed", response_model=DetailedHistoryResponse, tags=["History"])
def get_detailed_history(thread_id: str):
    """
    Get complete history with ALL metadata:
    - User query
    - Bot response
    - Tool used
    - Session ID
    - Timestamp
    
    Perfect for analytics and debugging!
    """
    if not chatbot:
        raise HTTPException(status_code=503)
    
    logger.info(f" Fetching detailed history for: {thread_id}")
    
    try:
        detailed = chatbot.get_detailed_history(thread_id)
        return DetailedHistoryResponse(**detailed)
    except Exception as e:
        logger.error(f" Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/{thread_id}", tags=["Statistics"])
def get_stats(thread_id: str):
    """Get session statistics"""
    if not chatbot:
        raise HTTPException(status_code=503)
    
    return chatbot.get_stats(thread_id)


@app.get("/sessions", tags=["Sessions"])
def list_sessions():
    """List all active sessions"""
    if not chatbot:
        raise HTTPException(status_code=503)
    
    sessions = chatbot.list_sessions()
    return {
        "total_sessions": len(sessions),
        "session_ids": sessions
    }


@app.delete("/clear/{thread_id}", tags=["Sessions"])
def clear_session(thread_id: str):
    """Clear session history"""
    if not chatbot:
        raise HTTPException(status_code=503)
    
    success = chatbot.clear_session(thread_id)
    if success:
        return {"message": f"Session {thread_id} cleared", "success": True}
    
    raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
