import gradio as gr
import requests

API_URL = "http://localhost:8000"
current_user = "gradio_user_001"


def respond(message, chat_history):
    """Handle chat with proper format detection"""
    if not message.strip():
        return "", chat_history
    
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "thread_id": current_user},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            bot_reply = data.get("response", "Error")
            tool = data.get("tool_used", "unknown")
            full_reply = f"{bot_reply}\n\n_Tool: {tool}_"
        else:
            full_reply = f"Error: {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        full_reply = "ERROR: Server not running!\n\nRun: python main.py"
    except Exception as e:
        full_reply = f"Error: {str(e)}"
    
    # Try modern format first
    try:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": full_reply})
    except:
        # Fallback to old format
        chat_history.append([message, full_reply])
    
    return "", chat_history


def get_stats():
    try:
        r = requests.get(f"{API_URL}/stats/{current_user}", timeout=10)
        if r.status_code == 200:
            d = r.json()
            t = f"Session: {d['session_id']}\n"
            t += f"Messages: {d['message_count']}\n\n"
            t += "Tools:\n"
            for tool, count in d.get('tools_used', {}).items():
                t += f"  {tool}: {count}\n"
            return t
        return f"Error: {r.status_code}"
    except Exception as e:
        return f"Error: {e}"


def get_full_history():
    try:
        r = requests.get(f"{API_URL}/history/{current_user}/detailed", timeout=10)
        if r.status_code == 200:
            d = r.json()
            t = f"Total: {d['total_messages']}\n\n"
            for m in d['conversation']:
                t += f"--- Message {m['message_id']} ---\n"
                t += f"You: {m['user_query']}\n"
                t += f"Bot: {m['bot_response'][:80]}...\n"
                t += f"Tool: {m['tool_used']}\n\n"
            return t
        return f"Error: {r.status_code}"
    except Exception as e:
        return f"Error: {e}"


def clear_all():
    try:
        requests.delete(f"{API_URL}/clear/{current_user}", timeout=10)
        return None, "Cleared!"
    except:
        return None, "Error"


# Simple interface using ChatInterface (most compatible)
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AI Chatbot\n\nTalk to an AI with specialized tools!")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=400)
            msg = gr.Textbox(label="Message", placeholder="Type here...")
            
            with gr.Row():
                submit = gr.Button("Send")
                clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            stats = gr.Textbox(label="Stats", lines=8)
            stats_btn = gr.Button("Show Stats")
            
            history = gr.Textbox(label="History", lines=8)
            history_btn = gr.Button("Show History")
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_all, None, [chatbot, stats])
    stats_btn.click(get_stats, None, stats)
    history_btn.click(get_full_history, None, history)

if __name__ == "__main__":
    print("="*50)
    print("LAUNCHING GRADIO")
    print("="*50)
    print("Make sure API is running: python main.py")
    print("="*50)
    demo.launch(server_port=7860)