import os
import requests
import re
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class NativeGoogleEmbeddings(Embeddings):
    def __init__(self, model: str = "gemini-embedding-2"):
        self.model = model.split("/")[-1]
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        self.url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:embedContent?key={self.api_key}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            payload = {
                "content": {
                    "parts": [{"text": text}]
                }
            }
            response = requests.post(self.url, json=payload, headers={"Content-Type": "application/json"})
            if response.status_code != 200:
                raise RuntimeError(f"Google API Error {response.status_code}: {response.text}")
                
            res_json = response.json()
            if "embedding" in res_json and "values" in res_json["embedding"]:
                embeddings.append(res_json["embedding"]["values"])
            else:
                raise KeyError(f"Unexpected response payload format: {res_json}")
                
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class SymbolGraph:
    """Extracts and maps where code symbols (Classes, Interfaces) are declared."""
    def __init__(self):
        # Maps symbol_name -> relative_file_path
        self.declarations = {}

    def extract_symbols(self, file_path: str, rel_path: str, ext: str):
        """Extracts class and interface declarations using robust structural signatures."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            
            found_symbols = []
            if ext in [".cs", ".java"]:
                # Matches patterns like: class PaymentRepository, interface IPaymentRepository
                found_symbols = re.findall(r'\b(?:class|interface)\s+([A-Za-z_][A-Za-z0-9_]*)', source_code)
            elif ext == ".py":
                # Matches patterns like: class PaymentProcessor:
                found_symbols = re.findall(r'\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b', source_code)
                
            for symbol_name in found_symbols:
                self.declarations[symbol_name] = rel_path
                
        except Exception as e:
            print(f"Error parsing symbols for {rel_path}: {e}")


class CodebaseIndexer:
    def __init__(self, repo_path: str, persist_directory: str = "./.chroma_code_db"):
        self.repo_path = repo_path
        self.persist_directory = persist_directory
        self.embeddings = NativeGoogleEmbeddings(model="gemini-embedding-2")
        self.symbol_graph = SymbolGraph()
        
    def index_repository(self):
        """Walks the repo, builds the structural symbol map, and indices vector embeddings."""
        documents = []
        extension_mappings = {
            ".cs": Language.CSHARP,
            ".py": Language.PYTHON,
            ".java": Language.JAVA
        }
        
        for root, dirs, files in os.walk(self.repo_path):
            if any(ignored in root for ignored in ["bin", "obj", ".git", "node_modules", ".venv"]):
                continue
                
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extension_mappings:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    
                    # 1. Harvest definitions into global symbol dictionary
                    self.symbol_graph.extract_symbols(file_path, rel_path, ext)
                    
                    # 2. Extract context text chunks
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        splitter = RecursiveCharacterTextSplitter.from_language(
                            language=extension_mappings[ext],
                            chunk_size=1000, 
                            chunk_overlap=100
                        )
                        
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
            print(f"Codebase indexing complete! Indexed {len(self.symbol_graph.declarations)} code symbols globally.")
        else:
            print("No matching source files found to index.")

    def query_relevant_code(self, query: str, top_k: int = 4) -> str:
        """Retrieves semantically matching chunks AND pulls linked references dynamically."""
        db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        results = db.similarity_search(query, k=top_k)
        
        context_blocks = {}
        referenced_files_to_pull = set()

        # Phase 1: Capture basic vector-matched blocks
        for doc in results:
            src = doc.metadata['source']
            if src not in context_blocks:
                context_blocks[src] = []
            context_blocks[src].append(doc.page_content)
            
            # Phase 2: Scan text for structural cross-references
            potential_symbols = re.findall(r'\b[I]?[A-Z][a-zA-Z0-9_]+\b', doc.page_content)
            for symbol in potential_symbols:
                if symbol in self.symbol_graph.declarations:
                    declaring_file = self.symbol_graph.declarations[symbol]
                    if declaring_file != src:
                        referenced_files_to_pull.add(declaring_file)

        # Phase 3: Inject missing dependency definitions into context
        for rel_path in referenced_files_to_pull:
            if rel_path not in context_blocks:
                full_path = os.path.join(self.repo_path, rel_path)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            # Pull definition header chunk safely
                            context_blocks[rel_path] = [f.read()[:1500] + "\n... [truncated reference block] ..."]
                    except Exception:
                        pass

        formatted_output = []
        for file, blocks in context_blocks.items():
            formatted_output.append(f"--- File: {file} ---")
            for block in blocks:
                formatted_output.append(block)
            formatted_output.append("\n")
            
        return "\n".join(formatted_output)









# import os
# import requests
# from langchain_chroma import Chroma
# from langchain_core.embeddings import Embeddings
# from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# class NativeGoogleEmbeddings(Embeddings):
#     def __init__(self, model: str = "gemini-embedding-2"):
#         self.model = model.split("/")[-1]
        
#         self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
#         if not self.api_key:
#             raise ValueError("API Key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
#         # Targets the production v1 endpoint
#         self.url = f"https://generativelanguage.googleapis.com/v1/models/{self.model}:embedContent?key={self.api_key}"

#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         if not texts:
#             return []
        
#         embeddings = []
#         for text in texts:
#             payload = {
#                 "content": {
#                     "parts": [{"text": text}]
#                 }
#             }
            
#             response = requests.post(self.url, json=payload, headers={"Content-Type": "application/json"})
            
#             if response.status_code != 200:
#                 raise RuntimeError(f"Google API Error {response.status_code}: {response.text}")
                
#             res_json = response.json()
#             if "embedding" in res_json and "values" in res_json["embedding"]:
#                 embeddings.append(res_json["embedding"]["values"])
#             else:
#                 raise KeyError(f"Unexpected response payload format: {res_json}")
                
#         return embeddings

#     def embed_query(self, text: str) -> list[float]:
#         return self.embed_documents([text])[0]


# class CodebaseIndexer:
#     def __init__(self, repo_path: str, persist_directory: str = "./.chroma_code_db"):
#         self.repo_path = repo_path
#         self.persist_directory = persist_directory
#         self.embeddings = NativeGoogleEmbeddings(model="gemini-embedding-2")
        
#     def index_repository(self):
#         """Walks the repo, chunks files based on extension, and builds the code vector store."""
#         documents = []
        
#         extension_mappings = {
#             ".cs": Language.CSHARP,
#             ".py": Language.PYTHON,
#             ".java": Language.JAVA
#         }
        
#         for root, dirs, files in os.walk(self.repo_path):
#             if any(ignored in root for ignored in ["bin", "obj", ".git", "node_modules", ".venv"]):
#                 continue
                
#             for file in files:
#                 ext = os.path.splitext(file)[1].lower()
#                 if ext in extension_mappings:
#                     file_path = os.path.join(root, file)
#                     try:
#                         with open(file_path, "r", encoding="utf-8") as f:
#                             content = f.read()
                            
#                         rel_path = os.path.relpath(file_path, self.repo_path)
                        
#                         splitter = RecursiveCharacterTextSplitter.from_language(
#                             language=extension_mappings[ext],
#                             chunk_size=1000, 
#                             chunk_overlap=100
#                         )
                        
#                         chunks = splitter.split_text(content)
#                         for idx, chunk in enumerate(chunks):
#                             doc = Document(
#                                 page_content=chunk,
#                                 metadata={
#                                     "source": rel_path,
#                                     "extension": ext,
#                                     "chunk_id": f"{rel_path}_{idx}"
#                                 }
#                             )
#                             documents.append(doc)
#                     except Exception as e:
#                         print(f"Skipping {file_path} due to read error: {e}")

#         if documents:
#             print(f"Building Code Vector DB with {len(documents)} code chunks...")
#             self.vector_store = Chroma.from_documents(
#                 documents=documents,
#                 embedding=self.embeddings,
#                 persist_directory=self.persist_directory
#             )
#             print("Codebase indexing complete!")
#         else:
#             print("No matching source files found to index.")

#     def query_relevant_code(self, query: str, top_k: int = 4) -> str:
#         """Retrieves only the code blocks relevant to the query to inject into context."""
#         db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
#         results = db.similarity_search(query, k=top_k)
        
#         context_blocks = []
#         for doc in results:
#             context_blocks.append(
#                 f"--- File: {doc.metadata['source']} ---\n{doc.page_content}\n"
#             )
#         return "\n".join(context_blocks)
