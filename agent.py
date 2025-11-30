"""
agent.py - Fast Agent with LangChain Buffer Memory
"""

import logging
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import ALL_TOOLS, positive_tool, negative_tool, student_marks_tool, crisis_tool
from memory import SessionManager

logger = logging.getLogger(__name__)

TOOL_SELECTOR_PROMPT = """Given a user message, select the most appropriate tool.

TOOLS:
1. positive_tool - For greetings, happy emotions, motivation
2. negative_tool - For sad emotions, struggles, problems
3. student_marks_tool - For academic queries about John, Sarah, Mike, Bob
4. crisis_tool - For self-harm or suicide mentions

USER MESSAGE: {message}

Respond with ONLY the tool name (e.g., "positive_tool"). No explanation."""


class ChatbotAgent:
    def __init__(self, nvidia_api_key: str):
        logger.info(" Initializing Chatbot Brain...")
        
        self.llm = ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            nvidia_api_key=nvidia_api_key,
            temperature=0.3,
            max_tokens=50
        )
        
        self.tool_selector = (
            ChatPromptTemplate.from_template(TOOL_SELECTOR_PROMPT)
            | self.llm
            | StrOutputParser()
        )
        
        self.tools_map = {
            "positive_tool": positive_tool,
            "negative_tool": negative_tool,
            "student_marks_tool": student_marks_tool,
            "crisis_tool": crisis_tool
        }
        
        # Using new LangChain memory
        self.session_manager = SessionManager()
        logger.info(" Agent with LangChain buffer memory initialized!")
    
    def chat(self, user_message: str, thread_id: str = "default") -> dict:
        """Process message with buffer memory"""
        logger.info(f" Processing: {user_message}")
        
        try:
            # Get conversation context from buffer
            context = self.session_manager.get_history(thread_id)
            
            # Add context to message if exists
            if context:
                # Only last 3 exchanges to keep it fast
                lines = context.split('\n')
                recent_context = '\n'.join(lines[-6:])
                full_message = f"{recent_context}\n\nHuman: {user_message}"
            else:
                full_message = user_message
            
            # Select tool
            tool_name = self._select_tool(user_message)
            logger.info(f" Selected: {tool_name}")
            
            # Execute tool
            bot_response = self._execute_tool(tool_name, user_message)
            
            # Save to buffer memory
            self.session_manager.save_interaction(thread_id, user_message, bot_response, tool_name)
            
            stats = self.session_manager.get_stats(thread_id)
            
            return {
                "success": True,
                "response": bot_response,
                "tool_used": tool_name,
                "message_count": stats["message_count"],
                "thread_id": thread_id
            }
        
        except Exception as e:
            logger.error(f" Error: {e}")
            return self._fallback(user_message, thread_id)
    
    def _select_tool(self, message: str) -> str:
        """LLM selects tool"""
        try:
            response = self.tool_selector.invoke({"message": message})
            tool_name = response.strip().lower()
            
            if tool_name in self.tools_map:
                return tool_name
            
            return self._keyword_match(message)
        
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return self._keyword_match(message)
    
    def _execute_tool(self, tool_name: str, message: str) -> str:
        """Execute tool"""
        tool = self.tools_map.get(tool_name)
        
        if not tool:
            return "I'm not sure how to help with that."
        
        try:
            if tool_name == "student_marks_tool":
                msg_lower = message.lower()
                for name in ["john", "sarah", "mike", "bob"]:
                    if name in msg_lower:
                        return tool(name)
                return "Which student? Available: John, Sarah, Mike, Bob"
            
            return tool(message)
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return "Sorry, I encountered an error."
    
    def _keyword_match(self, message: str) -> str:
        """Keyword-based fallback"""
        msg = message.lower()
        
        if any(w in msg for w in ["suicide", "kill myself", "self-harm", "end it", "want to die"]):
            return "crisis_tool"
        
        if any(w in msg for w in ["marks", "grades", "john", "sarah", "mike", "bob", "gpa"]):
            return "student_marks_tool"
        
        negative_words = [
            "sad", "down", "bad", "upset", "depressed", "heartbroken",
            "miserable", "terrible", "awful", "struggling", "hopeless"
        ]
        if any(w in msg for w in negative_words):
            return "negative_tool"
        
        positive_words = [
            "happy", "great", "good", "wonderful", "awesome",
            "hi", "hello", "hey", "morning", "afternoon", "evening"
        ]
        if any(w in msg for w in positive_words):
            return "positive_tool"
        
        return "positive_tool"
    
    def _fallback(self, message, thread_id):
        """Emergency fallback"""
        tool_name = self._keyword_match(message)
        response = self._execute_tool(tool_name, message)
        
        self.session_manager.save_interaction(thread_id, message, response, tool_name)
        stats = self.session_manager.get_stats(thread_id)
        
        return {
            "success": True,
            "response": response,
            "tool_used": tool_name,
            "message_count": stats["message_count"],
            "thread_id": thread_id
        }
    
    # Memory methods (updated for LangChain)
    def get_history(self, thread_id: str) -> str:
        return self.session_manager.get_history(thread_id)
    
    def get_detailed_history(self, thread_id: str) -> dict:
        return self.session_manager.get_detailed_history(thread_id)
    
    def get_stats(self, thread_id: str) -> dict:
        return self.session_manager.get_stats(thread_id)
    
    def clear_session(self, thread_id: str) -> bool:
        return self.session_manager.clear_session(thread_id)
    
    def list_sessions(self) -> list:
        return self.session_manager.list_all_sessions()
