import logging
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from tools import ALL_TOOLS
from memory import SessionManager

logger = logging.getLogger(__name__)


class ChatbotAgent:
    def __init__(self, nvidia_api_key: str):
        logger.info("Initializing Chatbot Brain...")
        
        self.llm = ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            nvidia_api_key=nvidia_api_key,
            temperature=0.7
        )
        logger.info("AI Model loaded!")
        
        self.tools = ALL_TOOLS
        logger.info(f"Loaded {len(self.tools)} tools!")
        
        self.session_manager = SessionManager()
        logger.info("Memory manager ready!")
        
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True  # Added this to capture tool usage
        )
        logger.info("Agent brain initialized!")
    
    def chat(self, user_message: str, thread_id: str = "default") -> dict:
        logger.info(f"Processing: {user_message[:50]}...")
        
        try:
            chat_history_str = self.session_manager.get_history(thread_id)
            
            if chat_history_str:
                full_message = f"Previous conversation:\n{chat_history_str}\n\nCurrent message: {user_message}"
            else:
                full_message = user_message
            
            result = self.agent_executor.invoke({"input": full_message})
            
            bot_response = result.get('output', 'Sorry, I could not understand that.')
            
            # Better tool detection
            tool_used = "unknown"
            if "intermediate_steps" in result and result["intermediate_steps"]:
                try:
                    # Get the first action (tool call)
                    first_step = result["intermediate_steps"][0]
                    if len(first_step) >= 1:
                        action = first_step[0]
                        # Try different ways to get tool name
                        if hasattr(action, 'tool'):
                            tool_used = action.tool
                        elif hasattr(action, 'name'):
                            tool_used = action.name
                        elif isinstance(action, tuple) and len(action) > 0:
                            if hasattr(action[0], 'tool'):
                                tool_used = action[0].tool
                        
                        logger.info(f"Detected tool from intermediate_steps: {tool_used}")
                except Exception as e:
                    logger.warning(f"Could not extract tool name: {e}")
            
            # If still unknown, try to detect from response content
            if tool_used == "unknown":
                response_lower = bot_response.lower()
                if "wonderful" in response_lower or "positive" in response_lower or "amazing" in response_lower:
                    tool_used = "positive_tool"
                elif "hear you" in response_lower or "valid" in response_lower or "difficult" in response_lower:
                    tool_used = "negative_tool"
                elif "academic report" in response_lower or "marks" in response_lower or "gpa" in response_lower:
                    tool_used = "student_marks_tool"
                elif "crisis" in response_lower or "988" in response_lower or "helpline" in response_lower:
                    tool_used = "crisis_tool"
            
            self.session_manager.save_interaction(
                thread_id=thread_id,
                user_msg=user_message,
                bot_response=bot_response,
                tool_name=tool_used
            )
            
            stats = self.session_manager.get_stats(thread_id)
            
            logger.info(f"Final tool used: {tool_used}")
            
            return {
                "success": True,
                "response": bot_response,
                "tool_used": tool_used,
                "message_count": stats["message_count"],
                "thread_id": thread_id
            }
        
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "response": f"Sorry, error: {str(e)}",
                "tool_used": "none",
                "message_count": 0,
                "thread_id": thread_id
            }
    
    def get_history(self, thread_id: str) -> str:
        return self.session_manager.get_history(thread_id)
    
    def get_stats(self, thread_id: str) -> dict:
        return self.session_manager.get_stats(thread_id)
    
    def clear_session(self, thread_id: str) -> bool:
        return self.session_manager.clear_session(thread_id)
    
    def list_sessions(self) -> list:
        return self.session_manager.list_all_sessions()