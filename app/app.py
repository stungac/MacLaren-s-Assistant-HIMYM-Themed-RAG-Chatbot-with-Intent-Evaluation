import streamlit as st # type: ignore
import base64
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -------------------- chose model from here -------------------- 
from models.openai_model import build_chat_chain
# from models.ollama_model import build_chat_chain

st.set_page_config(page_title="MacLaren's Assistant", layout="centered")

# Title and Background
st.markdown("""
    <div class='title-container'>
        <h1 style='text-align: center; color: #FFDD00; text-shadow: 2px 2px #000000;'>HIMYM Themed MacLaren's Asistant</h1>
    </div>
""", unsafe_allow_html=True)

# Background Image
image_path = "../app/maclarens.jpg"
with open(image_path, "rb") as img_file:
    image_base64 = base64.b64encode(img_file.read()).decode()

# CSS styles
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{image_base64}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        .background-overlay {{
            position: fixed; top: 0; bottom: 0; left: 50%;
            transform: translateX(-50%);
            width: 85%; max-width: 900px;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(8px);
            border-radius: 20px;
            z-index: 0;
        }}
        .title-container {{
            position: fixed; top: 70px; left: 50%;
            transform: translateX(-50%);
            z-index: 2;
            width: 80%; max-width: 850px;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
        }}
        .message-wrapper {{ padding-top: 120px; }}
        .chat-bubble {{
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            width: fit-content;
            max-width: 80%;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        }}
        .user-message {{ align-self: flex-end; background-color: rgba(255, 255, 255, 0.8); color: black; }}
        .bot-message {{ align-self: flex-start; background-color: rgba(0, 0, 0, 0.8); color: white; }}
        .message-container {{ display: flex; flex-direction: column; }}
    </style>
    <div class="background-overlay"></div>
    """, unsafe_allow_html=True
)

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

rag_chain, retriever = build_chat_chain()

query = st.chat_input(placeholder="Type something...")

if query:
    response = rag_chain.invoke({"input": query})
    st.session_state.chat_history.append({
        "user": query,
        "bot": response["answer"]
    })

    # Show predicted intent
    predicted_docs = retriever.get_relevant_documents(query)
    if predicted_docs:
        predicted_intent = predicted_docs[0].page_content.split("\n")[0].replace("Intent: ", "")
        st.success(f"Predicted Intent: {predicted_intent}")

st.markdown("""<div class="message-wrapper">""", unsafe_allow_html=True)
for chat in st.session_state.chat_history:
    st.markdown(f"""
        <div class="message-container">
            <div class="chat-bubble user-message">{chat['user']}</div>
            <div class="chat-bubble bot-message">{chat['bot']}</div>
        </div>
    """, unsafe_allow_html=True)
st.markdown("""</div>""", unsafe_allow_html=True)
