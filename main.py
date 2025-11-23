import logging
import sys
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from agent import ChatbotAgent

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

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    logger.error("NVIDIA_API_KEY not found!")
    sys.exit(1)

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

app = FastAPI(title="Chatbot API", version="1.0.0")

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
    logger.info("Starting server...")
    try:
        chatbot = ChatbotAgent(NVIDIA_API_KEY)
        logger.info("Server ready!")
        print("\n" + "="*60)
        print("SERVER IS RUNNING!")
        print("="*60)
        print("URL: http://localhost:8000")
        print("Docs: http://localhost:8000/docs")
        print("="*60 + "\n")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise

@app.get("/")
def home():
    return {"message": "Chatbot API", "version": "1.0.0", "status": "online"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(chatbot.list_sessions()) if chatbot else 0
    }

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not chatbot:
        raise HTTPException(status_code=503, detail="Not initialized")
    
    logger.info(f"Message from {request.thread_id}")
    
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
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{thread_id}")
def get_history(thread_id: str):
    if not chatbot:
        raise HTTPException(status_code=503)
    history = chatbot.get_history(thread_id)
    return {"thread_id": thread_id, "history": history}

@app.get("/stats/{thread_id}")
def get_stats(thread_id: str):
    if not chatbot:
        raise HTTPException(status_code=503)
    return chatbot.get_stats(thread_id)

@app.get("/sessions")
def list_sessions():
    if not chatbot:
        raise HTTPException(status_code=503)
    sessions = chatbot.list_sessions()
    return {"active_sessions": len(sessions), "session_ids": sessions}

@app.delete("/clear/{thread_id}")
def clear_session(thread_id: str):
    if not chatbot:
        raise HTTPException(status_code=503)
    success = chatbot.clear_session(thread_id)
    if success:
        return {"message": f"Session {thread_id} cleared", "success": True}
    raise HTTPException(status_code=404)

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")