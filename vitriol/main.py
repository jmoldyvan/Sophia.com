"""
Vitriol Content Ingestion Pipeline

This demo shows the event-driven architecture:
1. RedditDataSource polls for content (producer)
2. ContentQueue passes events asynchronously (channel/queue)
3. ContentConsumer processes incoming content (consumer)

Deduplication ensures the same post isn't processed twice.
"""

import asyncio
from mock_reddit_client import MockRedditClient
from content_queue import ContentQueue
from reddit_data_source import RedditDataSource
from content_consumer import ContentConsumer


async def main():
    print("=" * 60)
    print("  VITRIOL - Content Ingestion Pipeline Demo")
    print("=" * 60)
    print()
    
    # Initialize components
    reddit_client = MockRedditClient()
    content_queue = ContentQueue()
    
    data_source = RedditDataSource(
        client=reddit_client,
        queue=content_queue,
        subreddits=["politics", "news", "technology"],
        poll_interval_sec=3.0  # Poll every 3 seconds for demo
    )
    
    consumer = ContentConsumer(
        queue=content_queue,
        name="ToxicityScorer"
    )
    
    # Start consumer task (listens for items)
    consumer_task = asyncio.create_task(consumer.start_consuming())
    
    # Start producer task (polls Reddit)
    producer_task = asyncio.create_task(data_source.start_polling())
    
    # Run for 15 seconds then shut down
    print("[Main] Running for 15 seconds...")
    print()
    
    await asyncio.sleep(15)
    
    # Graceful shutdown
    print()
    print("=" * 60)
    print("  Shutting down...")
    print("=" * 60)
    
    data_source.stop()
    producer_task.cancel()
    
    try:
        await producer_task
    except asyncio.CancelledError:
        pass
    
    content_queue.close()
    await consumer_task
    
    # Final stats
    stats = data_source.get_stats()
    print()
    print("=" * 60)
    print("  Final Statistics")
    print("=" * 60)
    print(f"  Posts ingested:     {stats.total_ingested}")
    print(f"  Duplicates skipped: {stats.total_duplicates}")
    print(f"  Errors encountered: {stats.total_errors}")
    print(f"  Consumer processed: {consumer.get_processed_count()}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
