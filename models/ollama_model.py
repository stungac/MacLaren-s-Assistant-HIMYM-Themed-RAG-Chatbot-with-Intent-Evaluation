import os
import pandas as pd                                                             # type: ignore
from langchain_chroma import Chroma                                             # type: ignore
from langchain_core.prompts import ChatPromptTemplate                           # type: ignore
from langchain.chains.combine_documents import create_stuff_documents_chain     # type: ignore
from langchain.chains import create_retrieval_chain                             # type: ignore
from langchain.text_splitter import CharacterTextSplitter                       # type: ignore
from langchain_community.llms import Ollama                                     # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings                # type: ignore

# Data Preprocessing Function
def preprocess_data(data_path="../data/chatbot_dataset.xlsx"):
    df = pd.read_excel(data_path)
    df.dropna(subset=["Intent", "User Utterance", "Bot Response"], inplace=True)
    return df

# Chunking Function
def chunk_documents(df, chunk_size=500, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_chunks = []
    for _, row in df.iterrows():
        content = f"Intent: {row['Intent']}\nUser says: {row['User Utterance']}\nBot answer: {row['Bot Response']}"
        chunks = splitter.create_documents([content])
        all_chunks.extend(chunks)
    return all_chunks

# Load Vector Store 
def load_vectorstore(persist_path="../data/chroma_db_ollama", data_path="../data/chatbot_dataset.xlsx"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_path) and len(os.listdir(persist_path)) > 0:
        return Chroma(embedding_function=embeddings, persist_directory=persist_path)
    
    df = preprocess_data(data_path)
    documents = chunk_documents(df)
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_path
    )
    return vectorstore

# Build Chat Chain 
def build_chat_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 1}
    )

    try:
        llm = Ollama(model="mistral", temperature=0.4, num_predict=100)
    except Exception as e:
        print("Ollama couldn't be loaded. Does 'ollama serve' and 'ollama run mistral' work?")
        raise e

    system_prompt = (
        " Answer in 2-3 sentences max. "
        " You are “MacLaren's Assistant”, a How I Met Your Mother-themed chatbot. Respond to users with wit, references to HIMYM characters..."
        " If the user asks for alcohol, gently refuse and offer strawberry milk."
        " Based on the user's input, match the intent and retrieve similar examples.\n\nContext:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain, retriever
