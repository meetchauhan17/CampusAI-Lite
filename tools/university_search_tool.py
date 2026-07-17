import json
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from core.logger import logger
from core.ingestion import search_documents

# Keyword categories for query classification
CATEGORY_KEYWORDS = {
    "exams": ["exam", "timetable", "mid-sem", "midsem", "test", "theory", "hall ticket", "hallticket", "schedule"],
    "fees": ["fee", "tuition", "payment", "enrollment", "cost", "fine", "charges", "accounts", "dd", "demand draft"],
    "library": ["library", "book", "borrow", "circulation", "reading hall", "journal", "silence", "laptop"],
    "hostel": ["hostel", "mess", "dining", "curfew", "roll-call", "roll call", "warden", "disciplinary", "ragging", "night-out"],
    "academic-calendar": ["calendar", "vacation", "holiday", "diwali", "christmas", "mid-term", "term start", "commencement", "events"]
}

def classify_query(query: str) -> Optional[str]:
    """Classifies a search query into a document category using simple keyword matching."""
    lower_query = query.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lower_query for keyword in keywords):
            return category
    return None

def _execute_university_search(query: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Shared core execution logic for search.
    Classifies query, targets specific document chunks, and falls back if necessary.
    """
    category_detected = classify_query(query)
    
    if category_detected:
        logger.info("Search query classified into category: {}", category_detected)
        # Search targeted category first
        chunks = search_documents(query, top_k=top_k, category=category_detected)
        
        if chunks:
            return {
                "answer_chunks": chunks,
                "category_detected": category_detected,
                "confidence": 0.9
            }
        else:
            logger.info("No chunks found in category '{}'. Falling back to full-corpus search.", category_detected)
            # Fall back to searching everything
            all_chunks = search_documents(query, top_k=top_k, category=None)
            return {
                "answer_chunks": all_chunks,
                "category_detected": f"general (fallback from {category_detected})",
                "confidence": 0.5
            }
    else:
        logger.info("Query could not be classified. Performing full-corpus search.")
        all_chunks = search_documents(query, top_k=top_k, category=None)
        return {
            "answer_chunks": all_chunks,
            "category_detected": "general",
            "confidence": 0.7
        }

# LangChain args schema
class UniversitySearchInput(BaseModel):
    query: str = Field(..., description="The query string to search for in university documents.")

class UniversityInfoSearchTool(BaseTool):
    """
    LangChain BaseTool subclass wrapper for UniversityInfoSearchTool.
    """
    name: str = "UniversityInfoSearchTool"
    description: str = (
        "Search university documents (exams, fees, library, hostel, academic calendar) "
        "to answer student queries. Input should be a specific search query."
    )
    args_schema: Type[BaseModel] = UniversitySearchInput

    def _run(self, query: str) -> str:
        # Synchronous execution
        res = _execute_university_search(query)
        return json.dumps(res, default=str)

    async def _arun(self, query: str) -> str:
        # Asynchronous execution (delegates to sync in our mock)
        res = _execute_university_search(query)
        return json.dumps(res, default=str)


# PydanticAI Wrapper Function
def search_university_info(query: str) -> Dict[str, Any]:
    """
    Search university documents (exams, fees, library, hostel, academic calendar)
    to answer student queries.

    Args:
        query: The specific search query string.

    Returns:
        A dict containing search results:
        - answer_chunks: list of relevant document text chunks
        - category_detected: the category of documents searched
        - confidence: classification confidence score
    """
    return _execute_university_search(query)


if __name__ == "__main__":
    # Self-verification block
    print("--- UniversityInfoSearchTool Verification ---")
    test_queries = [
        "What is the fee for B.Tech semester 1?",
        "When is the Artificial Intelligence exam?",
        "How many library books can I borrow?",
        "What are the hostel curfew rules?"
    ]
    
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        result = _execute_university_search(q)
        print(f"Category Detected: {result['category_detected']} (Confidence: {result['confidence']})")
        print(f"Chunks Returned: {len(result['answer_chunks'])}")
        if result['answer_chunks']:
            print("First Chunk Text Snippet:", result['answer_chunks'][0]['text'][:150].replace('\n', ' '))
