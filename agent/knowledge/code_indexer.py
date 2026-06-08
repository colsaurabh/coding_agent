import os
import re
import pickle
import requests

from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)

def extract_methods(source_code: str, ext: str):

    if ext == ".py":
        return re.findall(
            r'def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
            source_code
        )

    return re.findall(
        r'(?:public|private|protected|internal)'
        r'(?:\s+(?:static|virtual|override|sealed|abstract))*'
        r'(?:\s+async)?'
        r'\s+[\w<>\[\],.?]+'
        r'\s+([A-Za-z_][A-Za-z0-9_]*)'
        r'\s*\(',
        source_code,
        re.MULTILINE
    )

def extract_method_name(chunk: str):

    match = re.search(
        r'(?:public|private|protected|internal).*?([A-Za-z_][A-Za-z0-9_]*)\s*\(',
        chunk,
        re.DOTALL
    )

    if match:
        return match.group(1)

    return "Unknown"



# ToDo: For other languages, we can implement similar logic or use language-specific parsers if available.
def extract_csharp_method_chunks(content: str):

    pattern = (
        r'(?:public|private|protected|internal)'
        r'(?:\s+(?:static|virtual|override|sealed|abstract))*'
        r'(?:\s+async)?'
        r'\s+[\w<>\[\],.?]+'
        r'\s+[A-Za-z_][A-Za-z0-9_]*'
        r'\s*\('
    )

    matches = list(
        re.finditer(
            pattern,
            content,
            re.MULTILINE
        )
    )

    if not matches:
        return [content]

    chunks = []

    for i, match in enumerate(matches):

        start = match.start()

        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(content)

        chunks.append(
            content[start:end].strip()
        )

    return chunks

# ============================================================
# Embeddings
# ============================================================

class NativeGoogleEmbeddings(Embeddings):

    def __init__(self, model: str = "gemini-embedding-2"):
        self.model = model.split("/")[-1]

        self.api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "API Key not found. "
                "Please set GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        self.url = (
            f"https://generativelanguage.googleapis.com/v1/models/"
            f"{self.model}:embedContent?key={self.api_key}"
        )

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        if not texts:
            return []

        embeddings = []

        for text in texts:

            payload = {
                "content": {
                    "parts": [
                        {
                            "text": text
                        }
                    ]
                }
            }

            response = requests.post(
                self.url,
                json=payload,
                headers={
                    "Content-Type": "application/json"
                }
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Google API Error "
                    f"{response.status_code}: "
                    f"{response.text}"
                )

            result = response.json()

            if (
                "embedding" in result
                and "values" in result["embedding"]
            ):
                embeddings.append(
                    result["embedding"]["values"]
                )
            else:
                raise KeyError(
                    f"Unexpected payload: {result}"
                )

        return embeddings

    def embed_query(
        self,
        text: str
    ) -> list[float]:

        return self.embed_documents([text])[0]


# ============================================================
# Symbol Graph
# ============================================================

class SymbolGraph:

    def __init__(self):

        self.class_index = {}
        self.method_index = {}

    def extract_symbols(
        self,
        file_path: str,
        rel_path: str,
        ext: str
    ):

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                source_code = f.read()

            found_classes = []

            if ext in [".cs", ".java"]:

                found_classes = re.findall(
                    r'\b(?:class|interface)\s+([A-Za-z_][A-Za-z0-9_]*)',
                    source_code
                )

            elif ext == ".py":

                found_classes = re.findall(
                    r'\bclass\s+([A-Za-z_][A-Za-z0-9_]*)',
                    source_code
                )

            for cls in found_classes:
                self.class_index[cls] = rel_path

            # methods = re.findall(
            #     r'(?:public|private|protected|internal)'
            #     r'\s+(?:async\s+)?'
            #     r'[\w<>\[\],]+'
            #     r'\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
            #     source_code
            # )
            methods = extract_methods(
                source_code,
                ext
            )
            print(f"{rel_path} -> methods found: {len(methods)}")
            

            for method in methods:
                self.method_index[method] = rel_path

        except Exception as e:

            print(
                f"Error parsing symbols "
                f"for {rel_path}: {e}"
            )

    def resolve_symbol(self, symbol: str):

        if symbol in self.class_index:
            return self.class_index[symbol]

        if symbol in self.method_index:
            return self.method_index[symbol]

        return None


# ============================================================
# Codebase Indexer
# ============================================================

class CodebaseIndexer:

    def __init__(
        self,
        repo_path: str,
        persist_directory: str = "./.chroma_code_db"
    ):

        self.repo_path = repo_path
        self.persist_directory = persist_directory

        self.embeddings = NativeGoogleEmbeddings(
            model="gemini-embedding-2"
        )

        self.symbol_graph = SymbolGraph()

    # --------------------------------------------------------

    def extract_metadata(self, content: str, ext: str):

        class_name = "Unknown"

        class_match = re.search(
            r'\bclass\s+([A-Za-z_][A-Za-z0-9_]*)',
            content
        )

        if class_match:
            class_name = class_match.group(1)

        # methods = re.findall(
        #     r'(?:public|private|protected|internal)'
        #     r'\s+(?:async\s+)?'
        #     r'[\w<>\[\],]+'
        #     r'\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
        #     content
        # )
        methods = extract_methods(
            content,
            ext
        )

        return {
            "class": class_name,
            "methods": ",".join(methods)
        }

    # --------------------------------------------------------

    def build_bm25(
        self,
        documents
    ):

        tokenized_docs = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        bm25 = BM25Okapi(tokenized_docs)

        os.makedirs(
            self.persist_directory,
            exist_ok=True
        )

        with open(
            os.path.join(
                self.persist_directory,
                "bm25.pkl"
            ),
            "wb"
        ) as f:

            pickle.dump(
                {
                    "bm25": bm25,
                    "documents": documents
                },
                f
            )

    # --------------------------------------------------------

    def bm25_search(
        self,
        query: str,
        k: int = 5
    ):

        bm25_path = os.path.join(
            self.persist_directory,
            "bm25.pkl"
        )

        if not os.path.exists(bm25_path):
            return []

        with open(
            bm25_path,
            "rb"
        ) as f:

            data = pickle.load(f)

        bm25 = data["bm25"]
        docs = data["documents"]

        scores = bm25.get_scores(
            query.lower().split()
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            docs[idx]
            for idx, _
            in ranked[:k]
        ]

    # --------------------------------------------------------

    def index_repository(self):

        documents = []

        extension_mappings = {
            ".cs": Language.CSHARP,
            ".py": Language.PYTHON,
            ".java": Language.JAVA
        }

        for root, dirs, files in os.walk(
            self.repo_path
        ):

            if any(
                ignored in root
                for ignored in [
                    "bin",
                    "obj",
                    ".git",
                    "node_modules",
                    ".venv"
                ]
            ):
                continue

            for file in files:

                ext = os.path.splitext(file)[1].lower()

                if ext not in extension_mappings:
                    continue

                file_path = os.path.join(
                    root,
                    file
                )

                rel_path = os.path.relpath(
                    file_path,
                    self.repo_path
                )

                print(f"Indexed: {rel_path}")

                self.symbol_graph.extract_symbols(
                    file_path,
                    rel_path,
                    ext
                )

                try:

                    with open(
                        file_path,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        content = f.read()

                    metadata = self.extract_metadata(
                        content,
                        ext
                    )

                    if ext == ".cs":

                        chunks = extract_csharp_method_chunks(
                            content
                        )
                        print(f"{rel_path} -> chunks created: {len(chunks)}")

                    else:

                        splitter = (
                            RecursiveCharacterTextSplitter
                            .from_language(
                                language=
                                extension_mappings[ext],
                                chunk_size=1000,
                                chunk_overlap=100
                            )
                        )

                        chunks = splitter.split_text(
                            content
                        )
                        print(f"{rel_path} -> chunks created: {len(chunks)}")

                    for idx, chunk in enumerate(
                        chunks
                    ):

                        method_name = extract_method_name(
                            chunk
                        )

                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source":
                                        rel_path,
                                    "extension":
                                        ext,
                                    "chunk_id":
                                        f"{rel_path}_{idx}",
                                    "class":
                                        metadata["class"],
                                    "method":
                                        method_name,
                                    "methods":
                                        metadata["methods"]
                                }
                            )
                        )

                except Exception as e:

                    print(
                        f"Skipping {file_path}: {e}"
                    )

        if not documents:

            print(
                "No source files found."
            )
            return

        print(
            f"Building Code Vector DB "
            f"with {len(documents)} chunks..."
        )

        self.vector_store = (
            Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=
                    self.persist_directory
            )
        )

        self.build_bm25(documents)

        print(
            f"Index complete.\n"
            f"Classes indexed: "
            f"{len(self.symbol_graph.class_index)}\n"
            f"Methods indexed: "
            f"{len(self.symbol_graph.method_index)}"
        )

    # --------------------------------------------------------

    def query_relevant_code(
        self,
        query: str,
        top_k: int = 8
    ) -> str:

        db = Chroma(
            persist_directory=
                self.persist_directory,
            embedding_function=
                self.embeddings
        )

        vector_results = (
            db.similarity_search(
                query,
                k=top_k
            )
        )

        bm25_results = (
            self.bm25_search(
                query,
                k=top_k
            )
        )

        results = []
        seen = set()

        for doc in (
            vector_results
            + bm25_results
        ):

            chunk_id = doc.metadata[
                "chunk_id"
            ]

            if chunk_id not in seen:

                seen.add(chunk_id)
                results.append(doc)

        context_blocks = {}

        referenced_files = set()

        for doc in results:

            source = doc.metadata[
                "source"
            ]

            if source not in context_blocks:
                context_blocks[source] = []

            context_blocks[source].append(
                doc.page_content
            )

            potential_symbols = re.findall(
                r'\b[I]?[A-Z][a-zA-Z0-9_]+\b',
                doc.page_content
            )

            for symbol in potential_symbols:

                declaring_file = (
                    self.symbol_graph
                    .resolve_symbol(
                        symbol
                    )
                )

                if (
                    declaring_file
                    and declaring_file != source
                ):
                    referenced_files.add(
                        declaring_file
                    )

        for rel_path in referenced_files:

            if rel_path in context_blocks:
                continue

            full_path = os.path.join(
                self.repo_path,
                rel_path
            )

            if not os.path.exists(
                full_path
            ):
                continue

            try:

                with open(
                    full_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    content = f.read()

                context_blocks[
                    rel_path
                ] = [
                    content[:1500]
                    + "\n...[truncated]..."
                ]

            except Exception:
                pass

        output = []

        for file_name, blocks in (
            context_blocks.items()
        ):

            output.append(
                f"--- File: {file_name} ---"
            )

            for block in blocks:
                output.append(block)

            output.append("")

        return "\n".join(output)