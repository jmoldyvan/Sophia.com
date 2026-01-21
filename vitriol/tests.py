"""
Vitriol Ingestion Pipeline - Test Suite

Tests cover:
- Functional correctness (ingestion, deduplication)
- Reliability (error handling)
- Maintainability (component isolation)
"""

import asyncio
import unittest
from model import ContentItem
from mock_reddit_client import MockRedditClient
from content_queue import ContentQueue
from reddit_data_source import RedditDataSource
from content_consumer import ContentConsumer


class TestContentItem(unittest.TestCase):
    """Test the ContentItem model."""
    
    def test_content_item_creation(self):
        """TC-01: Content items can be created with required fields."""
        item = ContentItem(
            id="test_123",
            source_id="reddit",
            subreddit="politics",
            title="Test Title",
            content="Test content",
            author="test_user",
            url="https://reddit.com/test",
            score=100,
            comment_count=50,
            created_at=1234567890.0
        )
        
        self.assertEqual(item.id, "test_123")
        self.assertEqual(item.subreddit, "politics")
        self.assertIsNotNone(item.ingested_at)
    
    def test_get_summary(self):
        """TC-02: Summary format is correct."""
        item = ContentItem(
            id="test_123",
            source_id="reddit",
            subreddit="news",
            title="Breaking News",
            content="Content here",
            author="reporter",
            url="https://reddit.com/test",
            score=500,
            comment_count=100,
            created_at=1234567890.0
        )
        
        summary = item.get_summary()
        self.assertIn("[news]", summary)
        self.assertIn("Breaking News", summary)
        self.assertIn("score: 500", summary)


class TestMockRedditClient(unittest.TestCase):
    """Test the mock Reddit client."""
    
    def test_fetch_posts_returns_correct_count(self):
        """TC-03: Fetch returns requested number of posts."""
        client = MockRedditClient()
        posts = client.fetch_posts("politics", limit=5)
        
        self.assertEqual(len(posts), 5)
        for post in posts:
            self.assertEqual(post.subreddit, "politics")
    
    def test_fetch_with_failure_can_raise(self):
        """TC-04: API failures are simulated correctly (reliability test)."""
        client = MockRedditClient()
        
        # With 100% failure rate, should always raise
        with self.assertRaises(RuntimeError) as context:
            client.fetch_with_possible_failure("test", failure_rate=1.0)
        
        self.assertIn("rate limit", str(context.exception))


class TestDeduplication(unittest.TestCase):
    """Test deduplication logic."""
    
    def test_duplicate_posts_are_skipped(self):
        """TC-05: Duplicate posts are not published to queue."""
        async def run_test():
            client = MockRedditClient()
            queue = ContentQueue()
            
            data_source = RedditDataSource(
                client=client,
                queue=queue,
                subreddits=["test"],
                poll_interval_sec=1.0
            )
            
            # Manually add a "seen" ID
            data_source.seen_ids.add("post_1")
            
            # Create a duplicate post
            duplicate = ContentItem(
                id="post_1",
                source_id="reddit",
                subreddit="test",
                title="Duplicate",
                content="Content",
                author="user",
                url="https://reddit.com/test",
                score=100,
                comment_count=10,
                created_at=1234567890.0
            )
            
            # Process it
            await data_source._process_posts([duplicate])
            
            # Should have been skipped
            self.assertEqual(data_source.total_duplicates, 1)
            self.assertEqual(data_source.total_ingested, 0)
        
        asyncio.run(run_test())


class TestAsyncPipeline(unittest.TestCase):
    """Test the async producer/consumer pipeline."""
    
    def test_queue_publish_and_consume(self):
        """TC-06: Items flow through queue correctly."""
        async def run_test():
            queue = ContentQueue()
            
            item = ContentItem(
                id="test_1",
                source_id="reddit",
                subreddit="test",
                title="Test",
                content="Content",
                author="user",
                url="https://reddit.com/test",
                score=100,
                comment_count=10,
                created_at=1234567890.0
            )
            
            await queue.publish(item)
            received = await queue.consume()
            
            self.assertEqual(received.id, "test_1")
            self.assertEqual(received.title, "Test")
        
        asyncio.run(run_test())
    
    def test_consumer_processes_all_items(self):
        """TC-07: Consumer processes all queued items."""
        async def run_test():
            queue = ContentQueue()
            consumer = ContentConsumer(queue, "TestConsumer")
            
            # Publish 3 items
            for i in range(3):
                item = ContentItem(
                    id=f"test_{i}",
                    source_id="reddit",
                    subreddit="test",
                    title=f"Post {i}",
                    content="Content",
                    author="user",
                    url="https://reddit.com/test",
                    score=100,
                    comment_count=10,
                    created_at=1234567890.0
                )
                await queue.publish(item)
            
            queue.close()
            
            # Start consumer
            await consumer.start_consuming()
            
            self.assertEqual(consumer.get_processed_count(), 3)
        
        asyncio.run(run_test())


class TestErrorHandling(unittest.TestCase):
    """Test error handling and reliability."""
    
    def test_api_errors_are_counted(self):
        """TC-08: API errors increment error counter (reliability)."""
        async def run_test():
            client = MockRedditClient()
            queue = ContentQueue()
            
            data_source = RedditDataSource(
                client=client,
                queue=queue,
                subreddits=["test"],
                poll_interval_sec=1.0
            )
            
            # Simulate an error
            data_source._handle_error("test", RuntimeError("Test error"))
            
            self.assertEqual(data_source.total_errors, 1)
        
        asyncio.run(run_test())


def run_tests():
    """Run all tests and print results."""
    print("=" * 60)
    print("  VITRIOL - Test Suite")
    print("=" * 60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestContentItem))
    suite.addTests(loader.loadTestsFromTestCase(TestMockRedditClient))
    suite.addTests(loader.loadTestsFromTestCase(TestDeduplication))
    suite.addTests(loader.loadTestsFromTestCase(TestAsyncPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    print("  Test Summary")
    print("=" * 60)
    print(f"  Tests run:    {result.testsRun}")
    print(f"  Failures:     {len(result.failures)}")
    print(f"  Errors:       {len(result.errors)}")
    print(f"  Success:      {result.wasSuccessful()}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_tests()
