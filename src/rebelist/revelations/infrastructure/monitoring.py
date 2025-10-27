"""Performance monitoring utilities for the RAG application."""

import time
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])


def monitor_performance(func: F) -> F:
    """Decorator to monitor function execution time."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"⏱️  {func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    return wrapper


class PerformanceTracker:
    """Simple performance tracker for RAG operations."""
    
    def __init__(self):
        self.timings: dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.timings[f"{operation}_start"] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        start_key = f"{operation}_start"
        if start_key not in self.timings:
            return 0.0
            
        duration = time.time() - self.timings[start_key]
        self.timings[operation] = duration
        del self.timings[start_key]
        
        print(f"⏱️  {operation}: {duration:.2f}s")
        return duration
    
    def get_summary(self) -> dict[str, float]:
        """Get performance summary."""
        return self.timings.copy()
