import asyncio
from content_queue import ContentQueue
from model import ContentItem

class ContentConsumer:
    """
    Consumer that processes content items from the queue.
    In the full system, this would pass items to the ToxicityScoringService.
    For now, it just logs received items.
    """
    
    def __init__(self, queue: ContentQueue, name: str = "Consumer-1"):
        self.queue = queue
        self.name = name
        self.processed_count = 0
        self.processed_items: list[ContentItem] = []
        self._running = False
    
    async def start_consuming(self):
        """Start consuming from the queue. Runs until queue is closed."""
        self._running = True
        print(f"[{self.name}] Started listening for content items...")
        print()
        
        while self._running:
            try:
                item = await self.queue.consume()
                if item is None:
                    if self.queue.is_closed:
                        break
                    continue
                await self._process_item(item)
            except asyncio.TimeoutError:
                if self.queue.is_closed:
                    break
                continue
        
        print(f"[{self.name}] Queue closed. Shutting down.")
    
    def stop(self):
        """Stop consuming."""
        self._running = False
    
    async def _process_item(self, item: ContentItem):
        """
        Process a single content item.
        This is where toxicity scoring would happen.
        """
        self.processed_count += 1
        self.processed_items.append(item)
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        print(f"[{self.name}] Processed #{self.processed_count}: {item.get_summary()}")
        
        # In full implementation:
        # score = await toxicity_scoring_service.score(item)
        # await repository.save(item, score)
    
    def get_processed_count(self) -> int:
        return self.processed_count
    
    def get_processed_items(self) -> list[ContentItem]:
        return self.processed_items.copy()
