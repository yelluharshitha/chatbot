"""
memory.py - LangChain-based Conversation Memory Manager
"""

import logging
from typing import List, Dict
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages multiple user sessions with LangChain memory"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        logger.info("SessionManager initialized (LangChain-based)")

    def get_or_create_session(self, thread_id: str) -> Dict:
        """Get existing session or create a new one"""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = {
                "memory": ConversationBufferMemory(return_messages=True),
                "tools_used": {},
                "tools_history": [],  # Track tool per human message
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "thread_id": thread_id
            }
            logger.info(f"New session created: {thread_id}")
        return self.sessions[thread_id]

    def save_interaction(self, thread_id: str, user_msg: str, bot_response: str, tool_name: str):
        """Save a conversation exchange for a specific session"""
        session = self.get_or_create_session(thread_id)
        memory: ConversationBufferMemory = session["memory"]

        # Save interaction using LangChain memory
        memory.save_context({"input": user_msg}, {"output": bot_response})

        # Update tool usage count
        session["tools_used"][tool_name] = session["tools_used"].get(tool_name, 0) + 1

        # Append tool to tools_history
        session["tools_history"].append(tool_name)

        # Update last active timestamp
        session["last_active"] = datetime.now().isoformat()

        logger.info(f"Interaction saved: {thread_id} | Tool used: {tool_name}")

    def get_history(self, thread_id: str) -> str:
        """Return conversation history as formatted string"""
        session = self.get_or_create_session(thread_id)
        memory: ConversationBufferMemory = session["memory"]
        messages = memory.load_memory_variables({}).get("history", [])

        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"AI: {msg.content}")
        return "\n".join(lines)

    def get_recent_messages(self, thread_id: str, limit: int = 5) -> List[Dict]:
        """Return last N messages with type info"""
        session = self.get_or_create_session(thread_id)
        memory: ConversationBufferMemory = session["memory"]
        messages = memory.load_memory_variables({}).get("history", [])

        recent = messages[-limit:] if limit else messages
        result = []
        for idx, msg in enumerate(recent, 1):
            result.append({
                "message_id": idx,
                "type": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content
            })
        return result

    def get_detailed_history(self, thread_id: str) -> dict:
        """Return complete conversation history with paired human + AI messages"""
        session = self.get_or_create_session(thread_id)
        memory: ConversationBufferMemory = session["memory"]
        messages = memory.load_memory_variables({}).get("history", [])
        tools_history = session.get("tools_history", [])

        conv_list = []
        human_index = 0
        idx = 1

        while human_index < len(tools_history):
            # Get human message
            human_msg = None
            ai_msg = None
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    human_msg = msg.content
                    messages.remove(msg)
                    break

            # Get next AI response
            for msg in messages:
                if isinstance(msg, AIMessage):
                    ai_msg = msg.content
                    messages.remove(msg)
                    break

            tool_used = tools_history[human_index] if human_index < len(tools_history) else "unknown"

            conv_list.append({
                "message_id": idx,
                "user_query": human_msg or "",
                "bot_response": ai_msg or "",
                "tool_used": tool_used,
                "session_id": thread_id,
                "timestamp": datetime.now().isoformat()
            })
            idx += 1
            human_index += 1

        return {
            "thread_id": thread_id,
            "session_id": thread_id,
            "total_messages": len(conv_list),
            "conversation": conv_list,
            "created_at": session["created_at"],
            "last_active": session["last_active"]
        }

    def get_stats(self, thread_id: str) -> Dict:
        """Get session statistics"""
        session = self.get_or_create_session(thread_id)
        memory: ConversationBufferMemory = session["memory"]
        messages = memory.load_memory_variables({}).get("history", [])
        return {
            "thread_id": thread_id,
            "message_count": len(messages),
            "tools_used": session["tools_used"],
            "created_at": session["created_at"],
            "last_active": session["last_active"]
        }

    def clear_session(self, thread_id: str) -> bool:
        """Clear all memory for a session"""
        if thread_id in self.sessions:
            del self.sessions[thread_id]
            logger.info(f"Session cleared: {thread_id}")
            return True
        return False

    def list_all_sessions(self) -> List[str]:
        """List all active sessions"""
        return list(self.sessions.keys())
