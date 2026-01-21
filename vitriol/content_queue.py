import asyncio
from model import ContentItem

class ContentQueue:
    """
    In-memory event queue using asyncio.Queue.
    
    This provides async producer/consumer pattern similar to Kafka topics.
    Can be replaced with Kafka/Redpanda producer later.
    """
    
    def __init__(self, max_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_size)
        self._closed = False
    
    async def publish(self, item: ContentItem):
        """Publish a content item to the queue."""
        if not self._closed:
            await self.queue.put(item)
    
    async def publish_all(self, items: list[ContentItem]):
        """Publish multiple items."""
        for item in items:
            await self.publish(item)
    
    async def consume(self) -> ContentItem | None:
        """
        Consume next item from queue.
        Returns None if queue is closed and empty.
        """
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if self._closed:
                return None
            raise
    
    def close(self):
        """Signal that no more items will be published."""
        self._closed = True
    
    @property
    def is_closed(self) -> bool:
        return self._closed and self.queue.empty()
