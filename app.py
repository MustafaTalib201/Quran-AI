import json
import gradio as gr
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# Load Quran
with open("quran.json", "r", encoding="utf-8") as f:
    verses = json.load(f)

texts = [v["text"] for v in verses]

# Load model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed(texts):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded)
    return model_output.last_hidden_state.mean(dim=1).numpy()

# Build index
embeddings = embed(texts)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Search function
def search(query):
    query_vec = embed([query])
    _, indices = index.search(query_vec, 3)
    return "\n\n".join(
        f"{verses[i]['surah']} [{verses[i]['ayah']}]: {verses[i]['text']}"
        for i in indices[0]
    )

# Gradio interface
gr.Interface(
    fn=search,
    inputs="text",
    outputs="text",
    title="📖 Quran RAG Bot",
    description="Ask questions and get verses from the Quran as answers."
).launch()
