"""
memory.py - LangChain Memory Implementation
"""

from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import os


class ChatMemory:
    """Chat memory manager using LangChain ConversationBufferMemory"""
    
    def __init__(self, api_key: str = None):
        """Initialize chat with LangChain memory"""
        
        # Set API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        # Initialize Claude LLM
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=self.api_key,
            max_tokens=1000,
            temperature=0.7
        )
        
        # Create LangChain ConversationBufferMemory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="response"
        )
        
        # Create conversation chain with memory
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    def chat(self, user_input: str) -> str:
        """
        Send message and get response with memory
        
        Args:
            user_input: User's message
            
        Returns:
            AI response string
        """
        response = self.conversation.predict(input=user_input)
        return response
    
    def get_chat_history(self) -> str:
        """
        Get full conversation history from memory
        
        Returns:
            Formatted chat history string
        """
        return self.memory.load_memory_variables({})["chat_history"]
    
    def clear_memory(self):
        """Clear all conversation history from memory"""
        self.memory.clear()
    
    def save_context(self, user_input: str, ai_response: str):
        """
        Manually save a conversation turn to memory
        
        Args:
            user_input: User's message
            ai_response: AI's response
        """
        self.memory.save_context(
            {"input": user_input},
            {"response": ai_response}
        )
    
    def get_memory_size(self) -> int:
        """Get number of messages in memory"""
        history = self.memory.load_memory_variables({})
        messages = history.get("chat_history", [])
        return len(messages)


# Example usage
if __name__ == "__main__":
    # Initialize chat with memory
    chat = ChatMemory()
    
    # Conversation 1
    print("User: Hi, my name is John")
    response1 = chat.chat("Hi, my name is John")
    print(f"AI: {response1}\n")
    
    # Conversation 2 (memory remembers name)
    print("User: What's my name?")
    response2 = chat.chat("What's my name?")
    print(f"AI: {response2}\n")
    
    # Conversation 3
    print("User: I like Python programming")
    response3 = chat.chat("I like Python programming")
    print(f"AI: {response3}\n")
    
    # Conversation 4 (memory remembers both name and interest)
    print("User: What do I like and what's my name?")
    response4 = chat.chat("What do I like and what's my name?")
    print(f"AI: {response4}\n")
    
    # Show memory stats
    print(f"Total messages in memory: {chat.get_memory_size()}")
    
    # View full history
    print("\n--- Full Chat History ---")
    print(chat.get_chat_history())
    
    # Clear memory
    print("\n--- Clearing Memory ---")
    chat.clear_memory()
    print(f"Messages after clear: {chat.get_memory_size()}")
