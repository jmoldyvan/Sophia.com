import asyncio
from dataclasses import dataclass
from mock_reddit_client import MockRedditClient
from content_queue import ContentQueue
from model import ContentItem

@dataclass
class IngestionStats:
    total_ingested: int
    total_duplicates: int
    total_errors: int
    unique_posts_seen: int


class RedditDataSource:
    """
    Data source that polls Reddit and publishes new content to the queue.
    Handles deduplication to avoid processing the same post twice.
    """
    
    def __init__(
        self,
        client: MockRedditClient,
        queue: ContentQueue,
        subreddits: list[str] = None,
        poll_interval_sec: float = 5.0
    ):
        self.client = client
        self.queue = queue
        self.subreddits = subreddits or ["politics", "news", "technology"]
        self.poll_interval_sec = poll_interval_sec
        
        # Track seen post IDs for deduplication
        self.seen_ids: set[str] = set()
        
        # Stats
        self.total_ingested = 0
        self.total_duplicates = 0
        self.total_errors = 0
        
        self._running = False
    
    async def start_polling(self):
        """Start the polling loop. Runs until stop() is called."""
        self._running = True
        print(f"[RedditDataSource] Starting polling for subreddits: {self.subreddits}")
        print(f"[RedditDataSource] Poll interval: {self.poll_interval_sec}s")
        print()
        
        while self._running:
            await self.poll()
            await asyncio.sleep(self.poll_interval_sec)
    
    def stop(self):
        """Stop the polling loop."""
        self._running = False
    
    async def poll(self):
        """Single poll cycle - fetch from all subreddits and publish new items."""
        print("[RedditDataSource] Polling...")
        
        for subreddit in self.subreddits:
            try:
                posts = self.client.fetch_with_possible_failure(subreddit)
                await self._process_posts(posts)
            except Exception as e:
                self._handle_error(subreddit, e)
        
        self._print_stats()
    
    async def _process_posts(self, posts: list[ContentItem]):
        """Process fetched posts - deduplicate and publish new ones."""
        for post in posts:
            if post.id not in self.seen_ids:
                # New post - publish to queue
                self.seen_ids.add(post.id)
                await self.queue.publish(post)
                self.total_ingested += 1
                print(f"  [NEW] {post.get_summary()}")
            else:
                # Duplicate - skip
                self.total_duplicates += 1
                print(f"  [DUP] Skipping {post.id}")
    
    def _handle_error(self, subreddit: str, error: Exception):
        """Handle API errors with logging."""
        self.total_errors += 1
        print(f"  [ERR] Failed to fetch r/{subreddit}: {error}")
        # In production: implement exponential backoff here
    
    def _print_stats(self):
        print(f"[Stats] Ingested: {self.total_ingested} | Duplicates: {self.total_duplicates} | Errors: {self.total_errors}")
        print()
    
    def get_stats(self) -> IngestionStats:
        return IngestionStats(
            total_ingested=self.total_ingested,
            total_duplicates=self.total_duplicates,
            total_errors=self.total_errors,
            unique_posts_seen=len(self.seen_ids)
        )
