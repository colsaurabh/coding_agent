import os
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from google import genai

class FunctionalKnowledgeEngine:
    def __init__(self, storage_path="./storage/vector_db"):
        self.ai_client = genai.Client()
        self.chroma_client = chromadb.PersistentClient(path=storage_path)
        
        from chromadb.utils.embedding_functions.google_embedding_function import (
            GoogleGeminiEmbeddingFunction,
        )
        self.embedding_fn = GoogleGeminiEmbeddingFunction()

        self.collection = self.chroma_client.get_or_create_collection(
            name="business_requirements",
            embedding_function=self.embedding_fn
        )

    def upload_business_pdf(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found at {file_path}")
            
        reader = PdfReader(file_path)
        chunks = []
        metadatas = []
        ids = []
        file_name = os.path.basename(file_path)
        
        print(f"Processing functional document: {file_name}...")
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text.strip():
                continue
            chunks.append(text)
            metadatas.append({"source": file_name, "page": page_num + 1})
            ids.append(f"{file_name}_page_{page_num + 1}")
            
        if chunks:
            self.collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)
            print(f"Successfully indexed {len(chunks)} pages.")
        else:
            print("Warning: No readable text found in PDF.")

    def query_business_rules(self, issue_description: str, limit: int = 3) -> str:
        results = self.collection.query(
            query_texts=[issue_description],
            n_results=limit
        )
        context_blocks = []
        if results and 'documents' in results and results['documents']:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                context_blocks.append(f"--- Document Reference ({meta['source']}, Page {meta['page']}) ---\n{doc}")
        return "\n\n".join(context_blocks)
