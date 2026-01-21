from dataclasses import dataclass
from time import time

@dataclass
class ContentItem:
    """Represents a piece of content ingested from a source."""
    id: str                 # Unique ID (Reddit post ID)
    source_id: str          # e.g., "reddit"
    subreddit: str          # e.g., "politics"
    title: str
    content: str
    author: str
    url: str
    score: int              # Reddit upvotes
    comment_count: int
    created_at: float       # Unix timestamp
    ingested_at: float = None
    
    def __post_init__(self):
        if self.ingested_at is None:
            self.ingested_at = time()
    
    def get_summary(self) -> str:
        return f"[{self.subreddit}] {self.title} (score: {self.score}, comments: {self.comment_count})"
