import os
import requests
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class NativeGoogleEmbeddings(Embeddings):
    def __init__(self, model: str = "text-embedding-004"):
        # Strip out any prefixes to ensure a clean model identifier
        self.model = model.split("/")[-1]
        
        # Pull key from standard environment variables
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        # Explicit production v1 REST endpoint string
        self.url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:embedContent?key={self.api_key}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        # Batch requests sequentially or as individual requests depending on standard API schema
        embeddings = []
        for text in texts:
            payload = {
                "content": {
                    "parts": [{"text": text}]
                },
                "taskType": "RETRIEVAL_DOCUMENT"
            }
            
            response = requests.post(self.url, json=payload, headers={"Content-Type": "application/json"})
            
            if response.status_code != 200:
                raise RuntimeError(f"Gemini API Error ({response.status_code}): {response.text}")
                
            data = response.json()
            embeddings.append(data["embedding"]["values"])
            
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        payload = {
            "content": {
                "parts": [{"text": text}]
            },
            "taskType": "RETRIEVAL_QUERY"
        }
        
        response = requests.post(self.url, json=payload, headers={"Content-Type": "application/json"})
        
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API Error ({response.status_code}): {response.text}")
            
        data = response.json()
        return data["embedding"]["values"]

class CodebaseIndexer:
    def __init__(self, repo_path: str, persist_directory: str = "./.chroma_code_db"):
        self.repo_path = repo_path
        self.persist_directory = persist_directory
        self.embeddings = NativeGoogleEmbeddings()
        
    def index_repository(self):
        """Walks the repo, chunks files based on extension, and builds the vector store."""
        documents = []
        
        # Supported extensions and their LangChain language mappings
        extension_mappings = {
            ".cs": Language.CSHARP,
            ".py": Language.PYTHON,
            ".java": Language.JAVA
        }
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip build/vcs directories
            if any(ignored in root for ignored in ["bin", "obj", ".git", "node_modules", ".venv"]):
                continue
                
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extension_mappings:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            
                        # Relative path makes it cleaner for the LLM to read reference tracks
                        rel_path = os.path.relpath(file_path, self.repo_path)
                        
                        # Initialize syntax-aware splitter for this specific language
                        splitter = RecursiveCharacterTextSplitter.from_language(
                            language=extension_mappings[ext],
                            chunk_size=1000, 
                            chunk_overlap=100
                        )
                        
                        # Generate chunks for this file
                        chunks = splitter.split_text(content)
                        for idx, chunk in enumerate(chunks):
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    "source": rel_path,
                                    "extension": ext,
                                    "chunk_id": f"{rel_path}_{idx}"
                                }
                            )
                            documents.append(doc)
                    except Exception as e:
                        print(f"Skipping {file_path} due to read error: {e}")

        if documents:
            print(f"Building Code Vector DB with {len(documents)} code chunks...")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("Codebase indexing complete!")
        else:
            print("No matching source files found to index.")

    def query_relevant_code(self, query: str, top_k: int = 4) -> str:
        """Retrieves only the code blocks relevant to the query to inject into context."""
        db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        results = db.similarity_search(query, k=top_k)
        
        context_blocks = []
        for doc in results:
            context_blocks.append(
                f"--- File: {doc.metadata['source']} ---\n{doc.page_content}\n"
            )
        return "\n".join(context_blocks)