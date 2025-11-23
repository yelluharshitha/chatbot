"""
memory.py - Conversation Memory Manager
"""

import logging
from typing import List, Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationMemory:
    """This class stores all conversations."""
    
    def __init__(self):
        self.messages: List[Tuple[str, str]] = []
        logger.info("Memory created - Ready to remember conversations!")
    
    def add_message(self, user_message: str, bot_response: str):
        """Save a new conversation exchange."""
        self.messages.append((user_message, bot_response))
        logger.info(f"Saved message #{len(self.messages)}")
    
    def get_history_text(self) -> str:
        """Get all conversations as readable text."""
        if not self.messages:
            return ""
        
        history_lines = []
        for user_msg, bot_msg in self.messages:
            history_lines.append(f"Human: {user_msg}")
            history_lines.append(f"AI: {bot_msg}")
        
        return "\n".join(history_lines)
    
    def get_message_count(self) -> int:
        """How many exchanges have happened?"""
        return len(self.messages)
    
    def clear(self):
        """Forget everything - start fresh"""
        self.messages = []
        logger.info("Memory cleared - Starting fresh!")


class SessionManager:
    """Manages multiple users at once."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        logger.info("SessionManager started - Ready for multiple users!")
    
    def get_or_create_session(self, thread_id: str) -> Dict:
        """Get a user's session, or create a new one if they're new."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = {
                "memory": ConversationMemory(),
                "tools_used": {},
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat()
            }
            logger.info(f"New session created for: {thread_id}")
        
        return self.sessions[thread_id]
    
    def save_interaction(self, thread_id: str, user_msg: str, bot_response: str, tool_name: str):
        """Save a conversation for a specific user."""
        session = self.get_or_create_session(thread_id)
        
        session["memory"].add_message(user_msg, bot_response)
        
        if tool_name not in session["tools_used"]:
            session["tools_used"][tool_name] = 0
        session["tools_used"][tool_name] += 1
        
        session["last_active"] = datetime.now().isoformat()
        
        logger.info(f"Saved interaction for {thread_id} (used: {tool_name})")
    
    def get_history(self, thread_id: str) -> str:
        """Get conversation history for a user"""
        session = self.get_or_create_session(thread_id)
        return session["memory"].get_history_text()
    
    def get_stats(self, thread_id: str) -> Dict:
        """Get statistics about a user's session"""
        session = self.get_or_create_session(thread_id)
        return {
            "thread_id": thread_id,
            "message_count": session["memory"].get_message_count(),
            "tools_used": session["tools_used"],
            "created_at": session["created_at"],
            "last_active": session["last_active"]
        }
    
    def clear_session(self, thread_id: str) -> bool:
        """Delete a user's session"""
        if thread_id in self.sessions:
            del self.sessions[thread_id]
            logger.info(f"Deleted session: {thread_id}")
            return True
        return False
    
    def list_all_sessions(self) -> List[str]:
        """Get list of all active users"""
        return list(self.sessions.keys())