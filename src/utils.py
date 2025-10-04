"""
Utility functions for Byte-builders_sentinel
Common helper functions used across the application.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# @algorithm Data Validation | Validates input data structure and content
def validate_event_data(event: Dict[str, Any]) -> bool:
    """
    Validate event data structure.
    
    Args:
        event: Event dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['timestamp', 'event_type']
    
    for field in required_fields:
        if field not in event:
            logger.warning(f"Missing required field: {field}")
            return False
    
    # Validate timestamp format
    try:
        datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        logger.warning(f"Invalid timestamp format: {event.get('timestamp')}")
        return False
    
    return True


# @algorithm File Processing | Handles file I/O operations for various data formats
def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of parsed JSON objects
    """
    data = []
    path = Path(file_path)
    
    if not path.exists():
        logger.warning(f"File not found: {file_path}")
        return data
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error at line {line_num}: {e}")
                        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
    
    return data


def save_jsonl_file(data: List[Dict[str, Any]], file_path: str) -> bool:
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} items to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")
        return False


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO formatted timestamp string
    """
    return datetime.now(timezone.utc).isoformat()


# @algorithm Configuration Management | Handles application configuration and settings
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load application configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "anomaly_threshold": 100,
        "max_events_per_batch": 1000,
        "dashboard_port": 5000,
        "log_level": "INFO"
    }
    
    path = Path(config_path)
    if path.exists():
        try:
            with open(path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Error loading config file: {e}")
    
    return default_config


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sentinel.log')
        ]
    )


class EventBuffer:
    """Buffer class for managing event streams"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.events = []
    
    def add_event(self, event: Dict[str, Any]) -> None:
        """Add event to buffer"""
        if validate_event_data(event):
            self.events.append(event)
            
            # Keep buffer size under limit
            if len(self.events) > self.max_size:
                self.events.pop(0)
    
    def get_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get events from buffer"""
        if limit:
            return self.events[-limit:]
        return self.events.copy()
    
    def clear(self) -> None:
        """Clear buffer"""
        self.events.clear()
    
    def size(self) -> int:
        """Get buffer size"""
        return len(self.events)