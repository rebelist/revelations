"""Hybrid search strategies for better document retrieval."""

from typing import Iterable, List
from rebelist.revelations.domain import ContextDocument


class HybridSearchStrategy:
    """Combines multiple search strategies for better retrieval quality."""
    
    def __init__(self, similarity_threshold: float = 0.3, max_docs: int = 10):
        self.similarity_threshold = similarity_threshold
        self.max_docs = max_docs
    
    def filter_and_rank(self, documents: List[ContextDocument], query: str) -> List[ContextDocument]:
        """Filter and rank documents based on multiple criteria."""
        if not documents:
            return documents
            
        # Filter by content length (avoid very short or very long chunks)
        filtered_docs = []
        for doc in documents:
            content_length = len(doc.content.strip())
            if 50 <= content_length <= 2000:  # Reasonable content length
                filtered_docs.append(doc)
        
        # If we have too few documents after filtering, include some longer ones
        if len(filtered_docs) < 3:
            for doc in documents:
                if doc not in filtered_docs and len(doc.content.strip()) > 20:
                    filtered_docs.append(doc)
        
        # Sort by title relevance (simple keyword matching)
        query_lower = query.lower()
        def title_relevance(doc: ContextDocument) -> int:
            title_lower = doc.title.lower()
            score = 0
            for word in query_lower.split():
                if word in title_lower:
                    score += 1
            return score
        
        # Sort by title relevance, then by content length
        filtered_docs.sort(key=lambda doc: (title_relevance(doc), len(doc.content)), reverse=True)
        
        return filtered_docs[:self.max_docs]
    
    def deduplicate_documents(self, documents: List[ContextDocument]) -> List[ContextDocument]:
        """Remove duplicate or very similar documents."""
        if not documents:
            return documents
            
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            # Create a simple hash of the content for deduplication
            content_hash = hash(doc.content.strip().lower())
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
