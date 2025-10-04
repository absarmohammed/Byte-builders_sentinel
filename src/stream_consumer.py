"""
Stream consumer for Byte-builders_sentinel
Connects to the streaming server and processes incoming events in real-time.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import SentinelSystem, process_events, detect_anomalies
from utils import EventBuffer, save_jsonl_file, get_current_timestamp

logger = logging.getLogger(__name__)


class StreamConsumer:
    """
    Real-time stream consumer that processes events from the streaming server
    and integrates with the sentinel system for anomaly detection.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.sentinel = SentinelSystem()
        self.event_buffer = EventBuffer(max_size=1000)
        self.processed_events = []
        self.anomalies = []
        self.running = False
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "processed_events": 0,
            "anomalies_detected": 0,
            "connection_time": None
        }
    
    # @algorithm Stream Processing | Real-time processing of streaming event data
    async def process_stream_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single streaming event and detect anomalies.
        
        Args:
            event: Raw event from stream
            
        Returns:
            Processed event or None if invalid
        """
        try:
            # Add metadata
            event["processed_timestamp"] = get_current_timestamp()
            event["source"] = "stream"
            
            # Add to buffer
            self.event_buffer.add_event(event)
            self.stats["total_events"] += 1
            
            # Process event using existing algorithms
            processed = process_events([event])
            if processed:
                processed_event = processed[0]
                self.processed_events.append(processed_event)
                self.stats["processed_events"] += 1
                
                # Check for anomalies in the processed event
                anomalies = detect_anomalies([processed_event])
                if anomalies:
                    self.anomalies.extend(anomalies)
                    self.stats["anomalies_detected"] += len(anomalies)
                    logger.warning(f"Anomaly detected: {anomalies[0]['anomaly_type']}")
                
                return processed_event
            
        except Exception as e:
            logger.error(f"Error processing stream event: {e}")
        
        return None
    
    # @algorithm Real-time Analytics | Calculates streaming metrics and patterns
    def calculate_stream_metrics(self) -> Dict[str, Any]:
        """
        Calculate real-time metrics from the stream.
        
        Returns:
            Dictionary of current metrics
        """
        recent_events = self.event_buffer.get_events(100)  # Last 100 events
        
        # Dataset distribution
        dataset_counts = {}
        for event in recent_events:
            dataset = event.get("dataset", "unknown")
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        # Anomaly rate
        anomaly_rate = (self.stats["anomalies_detected"] / max(1, self.stats["processed_events"])) * 100
        
        # Processing rate (events per second)
        processing_rate = 0.0
        if self.stats["connection_time"]:
            elapsed = (datetime.now() - self.stats["connection_time"]).total_seconds()
            if elapsed > 0:
                processing_rate = self.stats["total_events"] / elapsed
        
        return {
            "total_events_received": self.stats["total_events"],
            "events_processed": self.stats["processed_events"],
            "anomalies_detected": self.stats["anomalies_detected"],
            "anomaly_rate_percent": anomaly_rate,
            "processing_rate_per_second": processing_rate,
            "dataset_distribution": dataset_counts,
            "buffer_size": self.event_buffer.size(),
            "connection_uptime_seconds": (datetime.now() - self.stats["connection_time"]).total_seconds() if self.stats["connection_time"] else 0
        }
    
    async def connect_and_consume(self, callback: Optional[Callable] = None) -> None:
        """
        Connect to stream server and start consuming events.
        
        Args:
            callback: Optional callback function called for each processed event
        """
        try:
            logger.info(f"Connecting to stream server at {self.host}:{self.port}")
            reader, writer = await asyncio.open_connection(self.host, self.port)
            
            self.running = True
            self.stats["connection_time"] = datetime.now()
            logger.info("Connected to stream server")
            
            banner_received = False
            
            while self.running:
                line = await reader.readline()
                if not line:
                    logger.info("Server closed connection")
                    break
                
                try:
                    event = json.loads(line.decode('utf-8').strip())
                    
                    # Handle banner
                    if not banner_received and 'service' in event:
                        logger.info(f"Stream info: {event.get('datasets', [])} datasets available")
                        banner_received = True
                        continue
                    
                    # Process regular events
                    if banner_received:
                        processed_event = await self.process_stream_event(event)
                        
                        if processed_event and callback:
                            await callback(processed_event)
                        
                        # Print periodic stats
                        if self.stats["total_events"] % 50 == 0:
                            metrics = self.calculate_stream_metrics()
                            logger.info(f"Processed {metrics['total_events_received']} events, "
                                      f"{metrics['anomalies_detected']} anomalies detected, "
                                      f"Rate: {metrics['processing_rate_per_second']:.1f}/sec")
                
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    continue
            
            writer.close()
            await writer.wait_closed()
            
        except ConnectionRefusedError:
            logger.error(f"Could not connect to stream server at {self.host}:{self.port}")
            logger.info("Make sure the stream server is running with:")
            logger.info("  cd data/streaming-server && python server.py --loop")
        except Exception as e:
            logger.error(f"Stream connection error: {e}")
        finally:
            self.running = False
    
    def save_results(self, output_dir: Path) -> None:
        """Save processed events and anomalies to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all events (processed + anomalies)
        all_events = self.processed_events + self.anomalies
        events_file = output_dir / "events.jsonl"
        
        if save_jsonl_file(all_events, str(events_file)):
            logger.info(f"Saved {len(all_events)} events to {events_file}")
        
        # Save metrics
        metrics = self.calculate_stream_metrics()
        metrics_file = output_dir / "metrics.json"
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def stop(self) -> None:
        """Stop the stream consumer"""
        self.running = False
        logger.info("Stream consumer stopped")


async def event_callback(event: Dict[str, Any]) -> None:
    """Example callback function for processed events"""
    print(f"📊 Event processed: {event.get('dataset', 'unknown')} - {event.get('event_type', 'unknown')}")


async def main():
    """Main function for testing the stream consumer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stream Consumer for Sentinel System")
    parser.add_argument("--host", default="127.0.0.1", help="Stream server host")
    parser.add_argument("--port", type=int, default=8765, help="Stream server port")
    parser.add_argument("--duration", type=int, default=60, help="Duration to run (seconds)")
    parser.add_argument("--output-dir", default="evidence/output/test", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    consumer = StreamConsumer(args.host, args.port)
    
    try:
        # Run for specified duration
        consume_task = asyncio.create_task(
            consumer.connect_and_consume(event_callback)
        )
        
        if args.duration > 0:
            await asyncio.wait_for(consume_task, timeout=args.duration)
        else:
            await consume_task
            
    except asyncio.TimeoutError:
        logger.info(f"Stream consumption completed after {args.duration} seconds")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        consumer.stop()
        
        # Save results
        output_path = Path(args.output_dir)
        consumer.save_results(output_path)
        
        # Print final stats
        final_metrics = consumer.calculate_stream_metrics()
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        print("="*50)


if __name__ == "__main__":
    asyncio.run(main())