import os
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RAGPipeline:
    def __init__(self, documents):
        self.documents = documents
        self.dimension = len(self._get_embedding("test"))
        self.index = faiss.IndexFlatL2(self.dimension)
        self.doc_map = {}
        self._build_index()

    def _get_embedding(self, text):
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _build_index(self):
        embeddings = [self._get_embedding(doc) for doc in self.documents]
        self.index.add(np.array(embeddings))
        self.doc_map = {i: doc for i, doc in enumerate(self.documents)}

    def query(self, user_query, top_k=2):
        q_emb = self._get_embedding(user_query)
        D, I = self.index.search(np.array([q_emb]), top_k)
        context = "\n".join([self.doc_map[i] for i in I[0]])
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Use the context to answer the question."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_query}"}
            ]
        )
        return response.choices[0].message.content