import os
import glob
import threading
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from core.logger import logger

# Base Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "university_docs")
DB_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")

# Category mapping based on filename keywords
CATEGORY_MAPPING = {
    "exam": "exams",
    "fee": "fees",
    "library": "library",
    "hostel": "hostel",
    "calendar": "academic-calendar"
}

_client = None
_collection = None
_chroma_lock = threading.Lock()

def get_category_from_filename(filename: str) -> str:
    """Determine the category of a document based on its filename."""
    lower_filename = filename.lower()
    for keyword, category in CATEGORY_MAPPING.items():
        if keyword in lower_filename:
            return category
    return "general"

def get_chroma_collection():
    """Initialize ChromaDB client and retrieve or create the collection."""
    global _client, _collection
    if _collection is not None:
        return _collection

    with _chroma_lock:
        if _collection is not None:
            return _collection

        os.makedirs(DB_DIR, exist_ok=True)
        if _client is None:
            _client = chromadb.PersistentClient(path=DB_DIR)
        
        # Initialize sentence-transformers embedding function (runs locally)
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        _collection = _client.get_or_create_collection(
            name="university_docs",
            embedding_function=emb_fn
        )
        return _collection

def ingest_all_documents(refresh: bool = False) -> None:
    """
    Parses all documents in the data/university_docs/ folder using Docling,
    chunks them, and loads them into a local ChromaDB vector store.
    If refresh is True, clears the vector store before ingesting.
    """
    collection = get_chroma_collection()
    
    if refresh:
        logger.info("Refresh requested. Deleting existing chunks in vector store.")
        # ChromaDB delete all
        try:
            # Delete all documents by querying all IDs
            existing = collection.get()
            if existing and existing["ids"]:
                collection.delete(ids=existing["ids"])
                logger.info("Deleted {} existing entries.", len(existing["ids"]))
        except Exception as e:
            logger.warning("Error clearing collection: {}", e)

    # Find all docs in directory
    search_pattern = os.path.join(DOCS_DIR, "*")
    files = [f for f in glob.glob(search_pattern) if os.path.isfile(f) and not f.endswith(".gitkeep")]
    
    if not files:
        logger.warning("No files found in university_docs directory: {}", DOCS_DIR)
        return

    logger.info("Found {} files for ingestion.", len(files))
    
    converter = DocumentConverter()
    chunker = HybridChunker()

    for file_path in files:
        filename = os.path.basename(file_path)
        category = get_category_from_filename(filename)
        doc_id = os.path.splitext(filename)[0]
        
        logger.info("Processing file: {} (Category: {})", filename, category)
        
        try:
            # Parse file using Docling
            result = converter.convert(file_path)
            # Chunk the document structure
            chunks = list(chunker.chunk(result.document))
            
            logger.info("Parsed {} chunks from {}", len(chunks), filename)
            
            ids = []
            documents = []
            metadatas = []
            
            for idx, chunk in enumerate(chunks):
                # Retrieve text content
                text = chunk.text
                
                # Retrieve headings hierarchy
                section_title = ""
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    section_title = " > ".join(chunk.meta.headings)
                
                # Retrieve page number (best effort)
                page_number = 1
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        if hasattr(item, "prov") and item.prov:
                            p = item.prov[0]
                            if hasattr(p, "page_no") and p.page_no is not None:
                                page_number = p.page_no
                                break
                
                # Generate unique ID
                chunk_id = f"{doc_id}_{idx}"
                
                ids.append(chunk_id)
                documents.append(text)
                metadatas.append({
                    "doc_id": doc_id,
                    "source_file": filename,
                    "section_title": section_title,
                    "page_number": int(page_number),
                    "category": category
                })
                
            if ids:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                logger.info("Successfully loaded {} chunks from {} into vector store.", len(ids), filename)
                
        except Exception as e:
            logger.error("Failed to ingest file {}. Error: {}", filename, e)

def search_documents(query: str, top_k: int = 4, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search local vector store for query, optionally filtering by document category.
    """
    collection = get_chroma_collection()
    
    # Configure metadata filters if category is provided
    where_filter = None
    if category:
        where_filter = {"category": category}
        logger.info("Searching vector store for query: '{}' filtered by category: '{}'", query, category)
    else:
        logger.info("Searching vector store for query: '{}' across all categories", query)

    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )
        
        formatted_results = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "doc_id": meta.get("doc_id"),
                    "source_file": meta.get("source_file"),
                    "section_title": meta.get("section_title"),
                    "text": doc,
                    "page_number": meta.get("page_number"),
                    "category": meta.get("category"),
                    "distance": dist
                })
        
        logger.info("Found {} relevant chunks.", len(formatted_results))
        return formatted_results
        
    except Exception as e:
        logger.error("Failed to search vector store. Error: {}", e)
        return []
