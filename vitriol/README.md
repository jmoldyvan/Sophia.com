# Vitriol - Reddit Content Ingestion Pipeline

A prototype demonstrating event-driven content ingestion with async producer/consumer pattern.

## Project Structure

```
vitriol-python/
├── model.py              # ContentItem dataclass
├── mock_reddit_client.py # Simulates Reddit API
├── content_queue.py      # Async queue wrapper
├── reddit_data_source.py # Polling and deduplication
├── content_consumer.py   # Queue consumer
├── main.py               # Entry point
└── tests.py              # Test suite
```

## Running the Demo

```bash
python3 main.py
```

This runs a 15-second demo showing:
- Polling every 3 seconds from 3 subreddits
- Deduplication of repeated posts
- Async producer/consumer pipeline
- Final statistics

## Running Tests

```bash
python3 tests.py
```

Runs 8 test cases covering:
- Content item creation (TC-01, TC-02)
- Mock client behavior (TC-03, TC-04)
- Deduplication logic (TC-05)
- Queue operations (TC-06, TC-07)
- Error handling (TC-08)

## Architecture

```
MockRedditClient → RedditDataSource → ContentQueue → ContentConsumer
                         ↓
                   (deduplication)
```

The design follows an event-driven pattern where:
1. **Producer** (RedditDataSource) polls for content
2. **Queue** (ContentQueue) decouples producer from consumer
3. **Consumer** (ContentConsumer) processes items async

## Swapping to Real Reddit API

Replace `MockRedditClient` with a real client:

```python
# In main.py
from real_reddit_client import RealRedditClient
reddit_client = RealRedditClient(client_id="...", client_secret="...")
```

The rest of the code remains unchanged.
