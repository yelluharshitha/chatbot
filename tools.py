"""
tools.py - The Chatbot's Tools
"""

from langchain.tools import tool
import logging

logger = logging.getLogger(__name__)


@tool
def positive_tool(query: str) -> str:
    """This tool handles HAPPY and POSITIVE messages. When someone says I'm happy or motivate me, this tool responds."""
    logger.info(" Positive tool activated!")
    
    query_lower = query.lower()
    
    if "happy" in query_lower or "great" in query_lower:
        return " That's wonderful! Keep that positive energy going! Your happiness is inspiring!"
    
    elif "motivat" in query_lower or "inspire" in query_lower:
        return "You've got this! Every small step counts. Keep pushing forward - you're stronger than you think!"
    
    else:
        return " Stay positive! You're doing amazing. Keep that great attitude!"


@tool
def negative_tool(query: str) -> str:
    """This tool handles SAD and NEGATIVE emotions. When someone says I'm sad or I feel down, this tool responds with empathy."""
    logger.info(" Negative/empathy tool activated!")
    
    return """I hear you, and your feelings are completely valid. 

Remember:
- It's okay to not be okay sometimes
- This difficult moment will pass
- You've overcome challenges before, and you can do it again
- Small steps are still progress
- Take care of yourself today

Tomorrow is a new day with new possibilities. """


@tool
def student_marks_tool(student_name: str) -> str:
    """This tool looks up STUDENT GRADES. When someone asks What are John's marks, this tool finds the answer. Available students: John, Sarah, Mike, Bob."""
    logger.info(f"Student marks tool activated for: {student_name}")
    
    students = {
        "john": {
            "Math": 85,
            "Science": 92,
            "English": 78,
            "History": 88,
            "GPA": 3.6
        },
        "sarah": {
            "Math": 95,
            "Science": 89,
            "English": 91,
            "History": 87,
            "GPA": 3.9
        },
        "mike": {
            "Math": 72,
            "Science": 68,
            "English": 85,
            "History": 79,
            "GPA": 3.0
        },
        "bob": {
            "Math": 99,
            "Science": 77,
            "English": 88,
            "History": 99,
            "GPA": 4.0
        }
    }
    
    name = student_name.lower().strip()
    
    for student in students.keys():
        if student in name:
            name = student
            break
    
    if name not in students:
        return f"Student '{student_name}' not found.\n\nAvailable students: John, Sarah, Mike, Bob"
    
    marks = students[name]
    
    result = f"\n Academic Report for {name.title()}\n"
    result += "=" * 45 + "\n\n"
    
    for subject, score in marks.items():
        if subject != "GPA":
            result += f"  {subject:12s} : {score}/100\n"
    
    result += "\n" + "=" * 45 + "\n"
    result += f"  Overall GPA  : {marks['GPA']}/4.0\n"
    result += "=" * 45
    
    return result


@tool
def crisis_tool(query: str) -> str:
    """IMPORTANT: This tool provides crisis support. When someone mentions self-harm or suicide, this tool provides help resources."""
    logger.warning(" CRISIS TOOL ACTIVATED - Someone may need help!")
    
    return """
🆘 IMMEDIATE CRISIS SUPPORT RESOURCES
==========================================

If you're in crisis, please reach out RIGHT NOW:

📞 EMERGENCY HELPLINES:
  • 988 (US) - Suicide & Crisis Lifeline
    Available 24/7 - Call or Text
  
  • Text HOME to 741741
    Crisis Text Line - Free 24/7 support
  
  • 911 - For immediate emergency

🌍 INTERNATIONAL:
  Visit: https://www.iasp.info/resources/Crisis_Centres/

💙 YOU ARE NOT ALONE
Your life has value. Help is available RIGHT NOW.

Please reach out to:
  ✓ A trusted friend or family member
  ✓ Mental health professional
  ✓ Hospital emergency room
  ✓ Call 911 if in immediate danger

This feeling is temporary. Help is available. You matter.
==========================================
"""


ALL_TOOLS = [
    positive_tool,
    negative_tool,
    student_marks_tool,
    crisis_tool
]
