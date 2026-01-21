import random
from time import time
from model import ContentItem

class MockRedditClient:
    """
    Mock Reddit client that generates fake posts for testing.
    Can be swapped for RealRedditClient later.
    """
    
    MOCK_TITLES = [
        "Breaking: Major policy change announced today",
        "Scientists discover controversial new findings",
        "Heated debate erupts over proposed legislation",
        "Tech company faces backlash over privacy concerns",
        "Local community divided over new development",
        "Expert opinion sparks online controversy",
        "New study challenges conventional wisdom",
        "Government official makes surprising statement",
        "Industry leaders clash over regulations",
        "Viral post ignites fierce debate online"
    ]
    
    MOCK_CONTENT = [
        "This is a developing story with many perspectives...",
        "Critics argue that this approach is fundamentally flawed...",
        "Supporters claim the benefits outweigh the risks...",
        "The implications of this decision are far-reaching...",
        "Many are questioning the motives behind this move..."
    ]
    
    def __init__(self):
        self.post_counter = 0
    
    def fetch_posts(self, subreddit: str, limit: int = 5) -> list[ContentItem]:
        """
        Simulates fetching posts from a subreddit.
        Returns a mix of new and sometimes repeated posts to test deduplication.
        """
        posts = []
        for _ in range(limit):
            # 30% chance of repeat post to test deduplication
            is_repeat = random.random() < 0.3
            
            if is_repeat and self.post_counter > 0:
                post_id = f"post_{random.randint(1, self.post_counter)}"
            else:
                self.post_counter += 1
                post_id = f"post_{self.post_counter}"
            
            post = ContentItem(
                id=post_id,
                source_id="reddit",
                subreddit=subreddit,
                title=f"{random.choice(self.MOCK_TITLES)} #{post_id}",
                content=random.choice(self.MOCK_CONTENT),
                author=f"user_{random.randint(1000, 9999)}",
                url=f"https://reddit.com/r/{subreddit}/comments/{post_id}",
                score=random.randint(-10, 5000),
                comment_count=random.randint(0, 500),
                created_at=time() - random.randint(0, 86400)
            )
            posts.append(post)
        
        return posts
    
    def fetch_with_possible_failure(self, subreddit: str, failure_rate: float = 0.1) -> list[ContentItem]:
        """
        Simulates API failure for testing error handling.
        Raises exception with given probability.
        """
        if random.random() < failure_rate:
            raise RuntimeError("Reddit API rate limit exceeded")
        return self.fetch_posts(subreddit)
