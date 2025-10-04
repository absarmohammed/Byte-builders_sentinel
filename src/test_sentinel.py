"""
Test suite for Byte-builders_sentinel
Unit tests for all major components and algorithms.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.main import process_events, detect_anomalies, SentinelSystem
from src.utils import validate_event_data, load_jsonl_file, save_jsonl_file, EventBuffer


class TestEventProcessing:
    """Test event processing algorithms"""
    
    def test_process_events_basic(self):
        """Test basic event processing"""
        input_events = [
            {"type": "test", "data": {"value": 42}},
            {"type": "info", "data": {"message": "hello"}}
        ]
        
        result = process_events(input_events)
        
        assert len(result) == 2
        assert all(event["processed"] for event in result)
        assert result[0]["event_type"] == "test"
        assert result[1]["event_type"] == "info"
    
    def test_process_events_empty(self):
        """Test processing empty event list"""
        result = process_events([])
        assert result == []
    
    def test_detect_anomalies(self):
        """Test anomaly detection algorithm"""
        events = [
            {"data": {"value": 50}},  # Normal
            {"data": {"value": 150}}, # Anomaly (medium)
            {"data": {"value": 250}}, # Anomaly (high)
            {"data": {"other": "field"}} # No value field
        ]
        
        anomalies = detect_anomalies(events)
        
        assert len(anomalies) == 2
        assert anomalies[0]["severity"] == "medium"
        assert anomalies[1]["severity"] == "high"
        assert all(anomaly["event_type"] == "anomaly_detected" for anomaly in anomalies)


class TestDataValidation:
    """Test data validation functions"""
    
    def test_validate_event_data_valid(self):
        """Test validation of valid event data"""
        valid_event = {
            "timestamp": "2025-10-04T12:00:00Z",
            "event_type": "test",
            "data": {}
        }
        
        assert validate_event_data(valid_event)
    
    def test_validate_event_data_missing_fields(self):
        """Test validation with missing required fields"""
        invalid_event = {"data": {}}
        assert not validate_event_data(invalid_event)
        
        incomplete_event = {"timestamp": "2025-10-04T12:00:00Z"}
        assert not validate_event_data(incomplete_event)
    
    def test_validate_event_data_invalid_timestamp(self):
        """Test validation with invalid timestamp"""
        invalid_event = {
            "timestamp": "invalid-timestamp",
            "event_type": "test"
        }
        
        assert not validate_event_data(invalid_event)


class TestFileOperations:
    """Test file I/O operations"""
    
    def test_save_and_load_jsonl_file(self):
        """Test saving and loading JSONL files"""
        test_data = [
            {"id": 1, "name": "test1"},
            {"id": 2, "name": "test2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Test saving
            assert save_jsonl_file(test_data, temp_path)
            
            # Test loading
            loaded_data = load_jsonl_file(temp_path)
            assert loaded_data == test_data
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        result = load_jsonl_file("non_existent_file.jsonl")
        assert result == []


class TestEventBuffer:
    """Test event buffer functionality"""
    
    def test_event_buffer_basic(self):
        """Test basic event buffer operations"""
        buffer = EventBuffer(max_size=3)
        
        valid_event = {
            "timestamp": "2025-10-04T12:00:00Z",
            "event_type": "test",
            "data": {}
        }
        
        # Add events
        buffer.add_event(valid_event)
        assert buffer.size() == 1
        
        # Get events
        events = buffer.get_events()
        assert len(events) == 1
        assert events[0] == valid_event
    
    def test_event_buffer_overflow(self):
        """Test event buffer size limit"""
        buffer = EventBuffer(max_size=2)
        
        for i in range(3):
            event = {
                "timestamp": "2025-10-04T12:00:00Z",
                "event_type": f"test_{i}",
                "data": {"id": i}
            }
            buffer.add_event(event)
        
        # Should only keep the last 2 events
        assert buffer.size() == 2
        events = buffer.get_events()
        assert events[0]["data"]["id"] == 1
        assert events[1]["data"]["id"] == 2
    
    def test_event_buffer_invalid_events(self):
        """Test event buffer with invalid events"""
        buffer = EventBuffer()
        
        invalid_event = {"data": {}}  # Missing required fields
        buffer.add_event(invalid_event)
        
        assert buffer.size() == 0  # Invalid event should not be added


class TestSentinelSystem:
    """Test main sentinel system"""
    
    def test_sentinel_system_initialization(self):
        """Test system initialization"""
        system = SentinelSystem()
        
        assert system.events == []
        assert system.anomalies == []
    
    def test_sentinel_system_processing(self):
        """Test system data processing"""
        system = SentinelSystem()
        
        # Simulate loading data
        system.events = [
            {"type": "test", "data": {"value": 150}}
        ]
        
        system.process_data()
        
        # Check that events were processed
        assert len(system.events) > 0
        assert system.events[0]["processed"]
        
        # Check that anomalies were detected
        assert len(system.anomalies) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])