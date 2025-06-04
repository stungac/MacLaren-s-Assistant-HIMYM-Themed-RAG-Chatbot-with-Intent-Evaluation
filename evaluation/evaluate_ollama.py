import os
import pandas as pd                                                                 # type: ignore
from sklearn.metrics import classification_report, confusion_matrix                 # type: ignore
from sklearn.model_selection import train_test_split                                # type: ignore
from langchain_core.documents import Document                                       # type: ignore
from langchain_chroma import Chroma                                                 # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings                    # type: ignore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score # type: ignore

def preprocess_and_split(data_path="../data/chatbot_dataset.xlsx", test_size=0.2, val_size=0.1, random_state=42):
    df = pd.read_excel(data_path)
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['Intent'], random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size / (1 - test_size), stratify=train_val_df['Intent'], random_state=random_state)
    return train_df, val_df, test_df

def create_vectorstore_from_df(df, persist_path="../data/chroma_eval_ollama"):
    if os.path.exists(persist_path) and len(os.listdir(persist_path)) > 0:
        return Chroma(
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            persist_directory=persist_path
        )

    documents = [
        Document(page_content=f"Intent: {row['Intent']}\nUser says: {row['User Utterance']}\nBot answer: {row['Bot Response']}")
        for _, row in df.iterrows()
    ]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_path)
    return vectorstore

def evaluate_intent_accuracy(test_df, retriever):
    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        query = row['User Utterance']
        expected_intent = row['Intent']

        retrieved_docs = retriever.invoke(query)
        if retrieved_docs:
            first_doc = retrieved_docs[0].page_content
            predicted_intent = first_doc.split("\n")[0].replace("Intent: ", "").strip()
        else:
            predicted_intent = "None"

        y_true.append(expected_intent)
        y_pred.append(predicted_intent)

    print("\n---------- Classification Report (Per Class): ----------")
    print(classification_report(y_true, y_pred))

    print("\n---------- Confusion Matrix: ----------")
    print(confusion_matrix(y_true, y_pred))

    print("\n---------- Overall Metrics: ----------")
    print(f"Accuracy       : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1 Score : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro Recall   : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

if __name__ == "__main__":
    train_df, val_df, test_df = preprocess_and_split()
    vectorstore = create_vectorstore_from_df(train_df)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    evaluate_intent_accuracy(test_df, retriever)
