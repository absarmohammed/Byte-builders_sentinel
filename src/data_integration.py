"""
Advanced Data Integration Module for Retail Analytics
Handles multi-source data synchronization, correlation, and robust processing.
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataStreamConfig:
    """Configuration for a data stream"""
    name: str
    file_path: str
    expected_interval: int  # seconds between records
    required_fields: List[str]
    correlation_key: str  # field used for correlation (timestamp, station_id, etc.)
    max_delay_tolerance: int = 30  # maximum acceptable delay in seconds
    data_validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynchronizedEvent:
    """Represents a synchronized event with correlated data from multiple sources"""
    timestamp: datetime
    station_id: str
    primary_event: Dict[str, Any]
    correlated_events: Dict[str, List[Dict[str, Any]]]
    correlation_confidence: float
    data_quality_score: float
    missing_streams: List[str]
    anomaly_flags: List[str]


class DataQualityValidator:
    """Validates data quality and identifies issues"""
    
    @staticmethod
    def validate_event(event: Dict[str, Any], stream_config: DataStreamConfig) -> Tuple[bool, List[str]]:
        """
        Validate a single event against stream configuration.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        for field in stream_config.required_fields:
            if field not in event:
                issues.append(f"Missing required field: {field}")
            elif event[field] is None and field != "data":
                issues.append(f"Null value in required field: {field}")
        
        # Validate timestamp format
        try:
            if "timestamp" in event:
                datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            issues.append("Invalid timestamp format")
        
        # Validate data structure
        if "data" in event and event["data"] is not None:
            data = event["data"]
            
            # Stream-specific validations
            if stream_config.name == "pos_transactions":
                if isinstance(data, dict):
                    required_pos_fields = ["customer_id", "sku", "price"]
                    for field in required_pos_fields:
                        if field not in data:
                            issues.append(f"Missing POS data field: {field}")
                        elif field == "price" and (not isinstance(data[field], (int, float)) or data[field] < 0):
                            issues.append(f"Invalid price value: {data[field]}")
            
            elif stream_config.name == "queue_monitoring":
                if isinstance(data, dict):
                    if "customer_count" in data and (not isinstance(data["customer_count"], int) or data["customer_count"] < 0):
                        issues.append(f"Invalid customer_count: {data['customer_count']}")
                    if "average_dwell_time" in data and (not isinstance(data["average_dwell_time"], (int, float)) or data["average_dwell_time"] < 0):
                        issues.append(f"Invalid dwell_time: {data['average_dwell_time']}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def calculate_data_quality_score(event: Dict[str, Any], validation_issues: List[str]) -> float:
        """Calculate a data quality score (0.0 to 1.0)"""
        if not validation_issues:
            return 1.0
        
        # Deduct points for each issue type
        score = 1.0
        for issue in validation_issues:
            if "Missing required field" in issue:
                score -= 0.3
            elif "Invalid" in issue:
                score -= 0.2
            elif "Null value" in issue:
                score -= 0.1
            else:
                score -= 0.05
        
        return max(0.0, score)


# @algorithm Data Stream Synchronization | Synchronizes timestamped data across different sources
class DataStreamSynchronizer:
    """Synchronizes multiple data streams with different frequencies and patterns"""
    
    def __init__(self, time_window_seconds: int = 30):
        self.time_window_seconds = time_window_seconds
        self.stream_buffers = defaultdict(deque)  # Buffer for each stream
        self.synchronized_events = []
        self.last_sync_time = None
        self.data_validator = DataQualityValidator()
        
    def add_event_to_buffer(self, stream_name: str, event: Dict[str, Any]) -> None:
        """Add an event to the appropriate stream buffer"""
        try:
            timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            event["_parsed_timestamp"] = timestamp
            self.stream_buffers[stream_name].append(event)
            
            # Keep buffers manageable size
            if len(self.stream_buffers[stream_name]) > 1000:
                self.stream_buffers[stream_name].popleft()
                
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse timestamp for {stream_name}: {e}")
    
    def synchronize_by_time_window(self, target_timestamp: datetime) -> List[SynchronizedEvent]:
        """
        Synchronize events within a time window around target timestamp.
        
        Args:
            target_timestamp: The target time to synchronize around
            
        Returns:
            List of synchronized events
        """
        window_start = target_timestamp - timedelta(seconds=self.time_window_seconds // 2)
        window_end = target_timestamp + timedelta(seconds=self.time_window_seconds // 2)
        
        synchronized_events = []
        
        # Find events in time window for each stream
        windowed_events = {}
        for stream_name, buffer in self.stream_buffers.items():
            windowed_events[stream_name] = [
                event for event in buffer 
                if window_start <= event["_parsed_timestamp"] <= window_end
            ]
        
        # Group events by station and create synchronized events
        station_groups = defaultdict(lambda: defaultdict(list))
        
        for stream_name, events in windowed_events.items():
            for event in events:
                station_id = event.get("station_id", "unknown")
                station_groups[station_id][stream_name].append(event)
        
        # Create synchronized events for each station
        for station_id, stream_events in station_groups.items():
            if stream_events:  # Only create if we have events for this station
                sync_event = self._create_synchronized_event(
                    target_timestamp, station_id, stream_events
                )
                if sync_event:
                    synchronized_events.append(sync_event)
        
        return synchronized_events
    
    def _create_synchronized_event(
        self, 
        target_timestamp: datetime, 
        station_id: str, 
        stream_events: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[SynchronizedEvent]:
        """Create a synchronized event from multiple stream events"""
        
        # Determine primary event (POS transactions take priority)
        primary_event = None
        primary_stream = None
        
        priority_streams = ["pos_transactions", "product_recognition", "queue_monitoring", "rfid_readings"]
        
        for stream in priority_streams:
            if stream in stream_events and stream_events[stream]:
                # Find closest event to target timestamp
                closest_event = min(
                    stream_events[stream],
                    key=lambda x: abs((x["_parsed_timestamp"] - target_timestamp).total_seconds())
                )
                primary_event = closest_event
                primary_stream = stream
                break
        
        if not primary_event:
            return None
        
        # Calculate correlation confidence based on temporal proximity
        correlation_confidence = 1.0
        max_time_diff = 0
        
        for stream_name, events in stream_events.items():
            if events:
                closest_event = min(
                    events,
                    key=lambda x: abs((x["_parsed_timestamp"] - target_timestamp).total_seconds())
                )
                time_diff = abs((closest_event["_parsed_timestamp"] - target_timestamp).total_seconds())
                max_time_diff = max(max_time_diff, time_diff)
        
        # Reduce confidence based on maximum time difference
        if max_time_diff > 10:
            correlation_confidence *= (1.0 - min(max_time_diff / 60.0, 0.8))
        
        # Calculate overall data quality score
        all_events = []
        for events in stream_events.values():
            all_events.extend(events)
        
        quality_scores = []
        for event in all_events:
            _, issues = self.data_validator.validate_event(event, DataStreamConfig(
                name="generic",
                file_path="",
                expected_interval=5,
                required_fields=["timestamp", "station_id"],
                correlation_key="station_id"
            ))
            quality_scores.append(self.data_validator.calculate_data_quality_score(event, issues))
        
        data_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0
        
        # Identify missing streams
        expected_streams = ["pos_transactions", "rfid_readings", "queue_monitoring", "product_recognition"]
        missing_streams = [stream for stream in expected_streams if stream not in stream_events]
        
        # Detect anomaly flags
        anomaly_flags = []
        if primary_stream == "pos_transactions" and "rfid_readings" not in stream_events:
            anomaly_flags.append("potential_scan_avoidance")
        
        if correlation_confidence < 0.5:
            anomaly_flags.append("low_correlation_confidence")
        
        if data_quality_score < 0.7:
            anomaly_flags.append("poor_data_quality")
        
        return SynchronizedEvent(
            timestamp=target_timestamp,
            station_id=station_id,
            primary_event=primary_event,
            correlated_events=stream_events,
            correlation_confidence=correlation_confidence,
            data_quality_score=data_quality_score,
            missing_streams=missing_streams,
            anomaly_flags=anomaly_flags
        )


# @algorithm Cross-Stream Event Correlation | Correlates self-checkout activities with POS transactions, RFID and camera data
class CrossStreamCorrelator:
    """Advanced correlation engine for multi-stream retail data"""
    
    def __init__(self, correlation_window_seconds: int = 15):
        self.correlation_window = correlation_window_seconds
        self.correlation_cache = {}
        self.pattern_matcher = PatternMatcher()
    
    def correlate_checkout_activity(self, synchronized_events: List[SynchronizedEvent]) -> List[Dict[str, Any]]:
        """
        Correlate checkout activities across multiple data streams.
        
        Args:
            synchronized_events: List of synchronized events to correlate
            
        Returns:
            List of correlated activity patterns with confidence scores
        """
        correlated_activities = []
        
        # Group events by customer session (same customer_id within time window)
        customer_sessions = self._group_by_customer_sessions(synchronized_events)
        
        for customer_id, session_events in customer_sessions.items():
            if len(session_events) > 1:  # Only correlate if multiple events
                correlation = self._analyze_customer_session(customer_id, session_events)
                if correlation:
                    correlated_activities.append(correlation)
        
        # Also correlate by station activity patterns
        station_correlations = self._correlate_station_patterns(synchronized_events)
        correlated_activities.extend(station_correlations)
        
        return correlated_activities
    
    def _group_by_customer_sessions(self, events: List[SynchronizedEvent]) -> Dict[str, List[SynchronizedEvent]]:
        """Group synchronized events by customer sessions"""
        sessions = defaultdict(list)
        
        for event in events:
            # Extract customer_id from POS transactions
            customer_id = None
            if "pos_transactions" in event.correlated_events:
                for pos_event in event.correlated_events["pos_transactions"]:
                    if "data" in pos_event and pos_event["data"]:
                        customer_id = pos_event["data"].get("customer_id")
                        break
            
            if customer_id:
                sessions[customer_id].append(event)
        
        # Filter sessions by time proximity
        filtered_sessions = {}
        for customer_id, session_events in sessions.items():
            # Sort by timestamp
            session_events.sort(key=lambda x: x.timestamp)
            
            # Group into continuous sessions (max 10 minutes gap)
            current_session = []
            session_count = 0
            
            for event in session_events:
                if not current_session or (event.timestamp - current_session[-1].timestamp).total_seconds() <= 600:
                    current_session.append(event)
                else:
                    # Start new session
                    if len(current_session) > 1:
                        filtered_sessions[f"{customer_id}_session_{session_count}"] = current_session
                        session_count += 1
                    current_session = [event]
            
            # Add final session
            if len(current_session) > 1:
                filtered_sessions[f"{customer_id}_session_{session_count}"] = current_session
        
        return filtered_sessions
    
    def _analyze_customer_session(self, customer_id: str, events: List[SynchronizedEvent]) -> Optional[Dict[str, Any]]:
        """Analyze a customer session for patterns and anomalies"""
        
        if len(events) < 2:
            return None
        
        # Calculate session metrics
        session_start = min(events, key=lambda x: x.timestamp).timestamp
        session_end = max(events, key=lambda x: x.timestamp).timestamp
        session_duration = (session_end - session_start).total_seconds()
        
        # Analyze transaction patterns
        total_items = 0
        total_value = 0.0
        pos_events = []
        rfid_events = []
        recognition_events = []
        
        for event in events:
            if "pos_transactions" in event.correlated_events:
                pos_events.extend(event.correlated_events["pos_transactions"])
            if "rfid_readings" in event.correlated_events:
                rfid_events.extend(event.correlated_events["rfid_readings"])
            if "product_recognition" in event.correlated_events:
                recognition_events.extend(event.correlated_events["product_recognition"])
        
        # Calculate transaction metrics
        for pos_event in pos_events:
            if "data" in pos_event and pos_event["data"]:
                total_items += 1
                total_value += pos_event["data"].get("price", 0)
        
        # Detect potential anomalies
        anomaly_indicators = []
        confidence_factors = []
        
        # Check POS-RFID correlation
        if pos_events and not rfid_events:
            anomaly_indicators.append("no_rfid_for_pos_transactions")
            confidence_factors.append(0.8)
        
        # Check recognition accuracy
        low_accuracy_count = 0
        for rec_event in recognition_events:
            if "data" in rec_event and rec_event["data"]:
                accuracy = rec_event["data"].get("accuracy", 1.0)
                if accuracy < 0.7:
                    low_accuracy_count += 1
        
        if low_accuracy_count > len(recognition_events) * 0.3:  # More than 30% low accuracy
            anomaly_indicators.append("frequent_low_recognition_accuracy")
            confidence_factors.append(0.6)
        
        # Calculate overall correlation confidence
        correlation_confidence = statistics.mean([e.correlation_confidence for e in events])
        if confidence_factors:
            correlation_confidence *= statistics.mean(confidence_factors)
        
        return {
            "correlation_type": "customer_session",
            "customer_id": customer_id,
            "session_start": session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "session_duration_seconds": session_duration,
            "total_items": total_items,
            "total_value": total_value,
            "correlation_confidence": correlation_confidence,
            "anomaly_indicators": anomaly_indicators,
            "event_count": len(events),
            "stations_used": list(set(e.station_id for e in events))
        }
    
    def _correlate_station_patterns(self, events: List[SynchronizedEvent]) -> List[Dict[str, Any]]:
        """Correlate patterns at the station level"""
        station_correlations = []
        
        # Group by station
        station_events = defaultdict(list)
        for event in events:
            station_events[event.station_id].append(event)
        
        for station_id, station_event_list in station_events.items():
            if len(station_event_list) < 3:  # Need minimum events for pattern analysis
                continue
            
            # Sort by timestamp
            station_event_list.sort(key=lambda x: x.timestamp)
            
            # Analyze patterns
            correlation = self._analyze_station_activity_pattern(station_id, station_event_list)
            if correlation:
                station_correlations.append(correlation)
        
        return station_correlations
    
    def _analyze_station_activity_pattern(self, station_id: str, events: List[SynchronizedEvent]) -> Optional[Dict[str, Any]]:
        """Analyze activity patterns at a specific station"""
        
        # Calculate throughput metrics
        time_span = (events[-1].timestamp - events[0].timestamp).total_seconds()
        if time_span == 0:
            return None
        
        events_per_minute = len(events) / (time_span / 60.0)
        
        # Analyze data quality trends
        quality_scores = [e.data_quality_score for e in events]
        avg_quality = statistics.mean(quality_scores)
        quality_trend = "stable"
        
        if len(quality_scores) >= 3:
            first_half_avg = statistics.mean(quality_scores[:len(quality_scores)//2])
            second_half_avg = statistics.mean(quality_scores[len(quality_scores)//2:])
            
            if second_half_avg < first_half_avg - 0.1:
                quality_trend = "declining"
            elif second_half_avg > first_half_avg + 0.1:
                quality_trend = "improving"
        
        # Count anomaly flags
        anomaly_flag_counts = defaultdict(int)
        for event in events:
            for flag in event.anomaly_flags:
                anomaly_flag_counts[flag] += 1
        
        return {
            "correlation_type": "station_pattern",
            "station_id": station_id,
            "analysis_period_start": events[0].timestamp.isoformat(),
            "analysis_period_end": events[-1].timestamp.isoformat(),
            "events_per_minute": events_per_minute,
            "average_data_quality": avg_quality,
            "data_quality_trend": quality_trend,
            "anomaly_flag_summary": dict(anomaly_flag_counts),
            "correlation_confidence": statistics.mean([e.correlation_confidence for e in events])
        }


class PatternMatcher:
    """Identifies complex patterns in correlated data streams"""
    
    def __init__(self):
        self.known_patterns = {
            "scan_avoidance": self._detect_scan_avoidance_pattern,
            "barcode_switching": self._detect_barcode_switching_pattern,
            "weight_manipulation": self._detect_weight_manipulation_pattern,
            "system_abuse": self._detect_system_abuse_pattern
        }
    
    def _detect_scan_avoidance_pattern(self, events: List[SynchronizedEvent]) -> Dict[str, Any]:
        """Detect scan avoidance patterns"""
        # Implementation for scan avoidance detection
        return {"pattern": "scan_avoidance", "confidence": 0.0}
    
    def _detect_barcode_switching_pattern(self, events: List[SynchronizedEvent]) -> Dict[str, Any]:
        """Detect barcode switching patterns"""
        # Implementation for barcode switching detection
        return {"pattern": "barcode_switching", "confidence": 0.0}
    
    def _detect_weight_manipulation_pattern(self, events: List[SynchronizedEvent]) -> Dict[str, Any]:
        """Detect weight manipulation patterns"""
        # Implementation for weight manipulation detection
        return {"pattern": "weight_manipulation", "confidence": 0.0}
    
    def _detect_system_abuse_pattern(self, events: List[SynchronizedEvent]) -> Dict[str, Any]:
        """Detect system abuse patterns"""
        # Implementation for system abuse detection
        return {"pattern": "system_abuse", "confidence": 0.0}


# @algorithm Robust Data Processing | Handles missing, corrupt, or delayed data with adaptive strategies
class RobustDataProcessor:
    """Handles missing, corrupt, or delayed data with recovery strategies"""
    
    def __init__(self):
        self.missing_data_strategies = {
            "interpolation": self._interpolate_missing_data,
            "carry_forward": self._carry_forward_last_known,
            "default_values": self._use_default_values,
            "skip_incomplete": self._skip_incomplete_records
        }
        self.corruption_detectors = [
            self._detect_timestamp_corruption,
            self._detect_data_type_corruption,
            self._detect_value_range_corruption,
            self._detect_structural_corruption
        ]
        self.recovery_stats = defaultdict(int)
        
    def process_with_recovery(
        self, 
        raw_events: List[Dict[str, Any]], 
        stream_name: str,
        recovery_strategy: str = "adaptive"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process events with robust error handling and recovery.
        
        Args:
            raw_events: Raw events to process
            stream_name: Name of the data stream
            recovery_strategy: Strategy for handling issues
            
        Returns:
            (processed_events, processing_report)
        """
        
        processed_events = []
        processing_report = {
            "total_input_events": len(raw_events),
            "successful_events": 0,
            "corrupted_events": 0,
            "recovered_events": 0,
            "dropped_events": 0,
            "corruption_types": defaultdict(int),
            "recovery_methods": defaultdict(int)
        }
        
        for i, event in enumerate(raw_events):
            try:
                # First, detect corruption
                corruption_issues = self._detect_corruption(event, stream_name)
                
                if not corruption_issues:
                    # Event is clean
                    processed_events.append(event)
                    processing_report["successful_events"] += 1
                else:
                    # Event has issues - attempt recovery
                    processing_report["corrupted_events"] += 1
                    
                    for issue_type in corruption_issues:
                        processing_report["corruption_types"][issue_type] += 1
                    
                    if recovery_strategy == "adaptive":
                        recovery_method = self._select_optimal_recovery_method(corruption_issues, stream_name)
                    else:
                        recovery_method = recovery_strategy
                    
                    recovered_event = self._attempt_recovery(
                        event, corruption_issues, recovery_method, raw_events, i
                    )
                    
                    if recovered_event:
                        processed_events.append(recovered_event)
                        processing_report["recovered_events"] += 1
                        processing_report["recovery_methods"][recovery_method] += 1
                    else:
                        processing_report["dropped_events"] += 1
                        logger.warning(f"Dropped corrupted event from {stream_name}: {corruption_issues}")
                        
            except Exception as e:
                processing_report["dropped_events"] += 1
                logger.error(f"Failed to process event from {stream_name}: {e}")
        
        return processed_events, processing_report
    
    def _detect_corruption(self, event: Dict[str, Any], stream_name: str) -> List[str]:
        """Detect various types of data corruption"""
        corruption_issues = []
        
        for detector in self.corruption_detectors:
            issues = detector(event, stream_name)
            corruption_issues.extend(issues)
        
        return corruption_issues
    
    def _detect_timestamp_corruption(self, event: Dict[str, Any], stream_name: str) -> List[str]:
        """Detect timestamp-related corruption"""
        issues = []
        
        if "timestamp" not in event:
            issues.append("missing_timestamp")
        elif event["timestamp"] is None:
            issues.append("null_timestamp")
        else:
            try:
                ts = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                
                # Check for reasonable timestamp range (not too far in past/future)
                now = datetime.now()
                if abs((ts - now).total_seconds()) > 86400 * 365:  # More than 1 year difference
                    issues.append("unreasonable_timestamp")
                    
            except (ValueError, AttributeError):
                issues.append("invalid_timestamp_format")
        
        return issues
    
    def _detect_data_type_corruption(self, event: Dict[str, Any], stream_name: str) -> List[str]:
        """Detect data type corruption"""
        issues = []
        
        # Check station_id
        if "station_id" in event and not isinstance(event["station_id"], str):
            issues.append("invalid_station_id_type")
        
        # Stream-specific type checking
        if "data" in event and event["data"] is not None:
            data = event["data"]
            
            if stream_name == "pos_transactions" and isinstance(data, dict):
                if "price" in data and not isinstance(data["price"], (int, float)):
                    issues.append("invalid_price_type")
                if "weight_g" in data and not isinstance(data["weight_g"], (int, float)):
                    issues.append("invalid_weight_type")
            
            elif stream_name == "queue_monitoring" and isinstance(data, dict):
                if "customer_count" in data and not isinstance(data["customer_count"], int):
                    issues.append("invalid_customer_count_type")
        
        return issues
    
    def _detect_value_range_corruption(self, event: Dict[str, Any], stream_name: str) -> List[str]:
        """Detect values outside reasonable ranges"""
        issues = []
        
        if "data" in event and event["data"] is not None:
            data = event["data"]
            
            if stream_name == "pos_transactions" and isinstance(data, dict):
                price = data.get("price", 0)
                if isinstance(price, (int, float)) and (price < 0 or price > 100000):
                    issues.append("unreasonable_price_range")
                
                weight = data.get("weight_g", 0)
                if isinstance(weight, (int, float)) and (weight < 0 or weight > 50000):
                    issues.append("unreasonable_weight_range")
            
            elif stream_name == "queue_monitoring" and isinstance(data, dict):
                customer_count = data.get("customer_count", 0)
                if isinstance(customer_count, int) and (customer_count < 0 or customer_count > 100):
                    issues.append("unreasonable_customer_count")
                
                dwell_time = data.get("average_dwell_time", 0)
                if isinstance(dwell_time, (int, float)) and (dwell_time < 0 or dwell_time > 3600):
                    issues.append("unreasonable_dwell_time")
        
        return issues
    
    def _detect_structural_corruption(self, event: Dict[str, Any], stream_name: str) -> List[str]:
        """Detect structural corruption in data"""
        issues = []
        
        if not isinstance(event, dict):
            issues.append("non_dict_event")
            return issues
        
        # Check for completely empty events
        if not event or all(v is None for v in event.values()):
            issues.append("empty_event")
        
        # Check data field structure
        if "data" in event:
            data = event["data"]
            if data is not None and not isinstance(data, dict):
                issues.append("invalid_data_structure")
        
        return issues
    
    def _select_optimal_recovery_method(self, corruption_issues: List[str], stream_name: str) -> str:
        """Select the best recovery method based on corruption type"""
        
        # Priority-based selection
        if any("timestamp" in issue for issue in corruption_issues):
            return "interpolation"
        elif any("type" in issue for issue in corruption_issues):
            return "default_values"
        elif any("range" in issue for issue in corruption_issues):
            return "carry_forward"
        else:
            return "skip_incomplete"
    
    def _attempt_recovery(
        self, 
        event: Dict[str, Any], 
        corruption_issues: List[str], 
        recovery_method: str,
        all_events: List[Dict[str, Any]],
        current_index: int
    ) -> Optional[Dict[str, Any]]:
        """Attempt to recover a corrupted event"""
        
        if recovery_method not in self.missing_data_strategies:
            return None
        
        try:
            return self.missing_data_strategies[recovery_method](
                event, corruption_issues, all_events, current_index
            )
        except Exception as e:
            logger.error(f"Recovery failed with method {recovery_method}: {e}")
            return None
    
    def _interpolate_missing_data(
        self, 
        event: Dict[str, Any], 
        issues: List[str], 
        all_events: List[Dict[str, Any]], 
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Interpolate missing data from neighboring events"""
        
        recovered_event = event.copy()
        
        # Fix timestamp through interpolation
        if any("timestamp" in issue for issue in issues):
            prev_event = None
            next_event = None
            
            # Find previous valid event
            for i in range(index - 1, -1, -1):
                if "timestamp" in all_events[i] and all_events[i]["timestamp"]:
                    try:
                        datetime.fromisoformat(all_events[i]["timestamp"].replace("Z", "+00:00"))
                        prev_event = all_events[i]
                        break
                    except:
                        continue
            
            # Find next valid event
            for i in range(index + 1, len(all_events)):
                if "timestamp" in all_events[i] and all_events[i]["timestamp"]:
                    try:
                        datetime.fromisoformat(all_events[i]["timestamp"].replace("Z", "+00:00"))
                        next_event = all_events[i]
                        break
                    except:
                        continue
            
            if prev_event and next_event:
                # Interpolate timestamp
                prev_ts = datetime.fromisoformat(prev_event["timestamp"].replace("Z", "+00:00"))
                next_ts = datetime.fromisoformat(next_event["timestamp"].replace("Z", "+00:00"))
                
                interpolated_ts = prev_ts + (next_ts - prev_ts) / 2
                recovered_event["timestamp"] = interpolated_ts.isoformat()
            
            elif prev_event:
                # Estimate based on typical interval (5 seconds for most streams)
                prev_ts = datetime.fromisoformat(prev_event["timestamp"].replace("Z", "+00:00"))
                estimated_ts = prev_ts + timedelta(seconds=5)
                recovered_event["timestamp"] = estimated_ts.isoformat()
        
        return recovered_event
    
    def _carry_forward_last_known(
        self, 
        event: Dict[str, Any], 
        issues: List[str], 
        all_events: List[Dict[str, Any]], 
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Carry forward last known good values"""
        
        recovered_event = event.copy()
        
        # Find the last valid event
        for i in range(index - 1, -1, -1):
            prev_event = all_events[i]
            
            # Copy missing or corrupted fields
            for field in ["timestamp", "station_id", "status"]:
                if field not in recovered_event or recovered_event[field] is None:
                    if field in prev_event and prev_event[field] is not None:
                        recovered_event[field] = prev_event[field]
            
            # Handle data field
            if "data" not in recovered_event or not recovered_event["data"]:
                if "data" in prev_event and prev_event["data"]:
                    recovered_event["data"] = prev_event["data"].copy()
            
            break
        
        return recovered_event
    
    def _use_default_values(
        self, 
        event: Dict[str, Any], 
        issues: List[str], 
        all_events: List[Dict[str, Any]], 
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Use reasonable default values for missing/corrupted data"""
        
        recovered_event = event.copy()
        
        # Default timestamp to current time if missing
        if "timestamp" not in recovered_event or not recovered_event["timestamp"]:
            recovered_event["timestamp"] = datetime.now().isoformat()
        
        # Default station_id
        if "station_id" not in recovered_event or not recovered_event["station_id"]:
            recovered_event["station_id"] = "UNKNOWN"
        
        # Default status
        if "status" not in recovered_event or not recovered_event["status"]:
            recovered_event["status"] = "Active"
        
        # Handle data field with stream-specific defaults
        if "data" not in recovered_event or not recovered_event["data"]:
            recovered_event["data"] = {}
        
        return recovered_event
    
    def _skip_incomplete_records(
        self, 
        event: Dict[str, Any], 
        issues: List[str], 
        all_events: List[Dict[str, Any]], 
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Skip records that cannot be recovered"""
        return None  # Indicates the event should be dropped


# @algorithm Data Integration Pipeline | Orchestrates the complete data integration process
class DataIntegrationPipeline:
    """Main pipeline orchestrating the complete data integration process"""
    
    def __init__(self, data_directory: str = "data/input"):
        self.data_directory = Path(data_directory)
        self.synchronizer = DataStreamSynchronizer(time_window_seconds=30)
        self.correlator = CrossStreamCorrelator(correlation_window_seconds=15)
        self.processor = RobustDataProcessor()
        
        # Configure data streams
        self.stream_configs = {
            "pos_transactions": DataStreamConfig(
                name="pos_transactions",
                file_path="pos_transactions.jsonl",
                expected_interval=5,
                required_fields=["timestamp", "station_id", "status", "data"],
                correlation_key="station_id"
            ),
            "rfid_readings": DataStreamConfig(
                name="rfid_readings", 
                file_path="rfid_readings.jsonl",
                expected_interval=5,
                required_fields=["timestamp", "station_id", "status", "data"],
                correlation_key="station_id"
            ),
            "queue_monitoring": DataStreamConfig(
                name="queue_monitoring",
                file_path="queue_monitoring.jsonl", 
                expected_interval=5,
                required_fields=["timestamp", "station_id", "status", "data"],
                correlation_key="station_id"
            ),
            "product_recognition": DataStreamConfig(
                name="product_recognition",
                file_path="product_recognition.jsonl",
                expected_interval=1,
                required_fields=["timestamp", "station_id", "status", "data"],
                correlation_key="station_id"
            ),
            "inventory_data": DataStreamConfig(
                name="inventory_data",
                file_path="inventory_data.jsonl",
                expected_interval=600,  # 10 minutes
                required_fields=["timestamp", "data"],
                correlation_key="product_id"
            )
        }
        
        self.processing_stats = {}
    
    async def process_all_streams(self) -> Dict[str, Any]:
        """
        Process all data streams with full integration pipeline.
        
        Returns:
            Comprehensive integration results including synchronized events and correlations
        """
        
        logger.info("Starting data integration pipeline...")
        
        # Step 1: Load and process each stream
        stream_data = {}
        processing_reports = {}
        
        for stream_name, config in self.stream_configs.items():
            file_path = self.data_directory / config.file_path
            
            if file_path.exists():
                logger.info(f"Processing {stream_name} from {file_path}")
                
                # Load raw data
                raw_events = self._load_jsonl_file(file_path)
                
                # Process with robust error handling
                processed_events, report = self.processor.process_with_recovery(
                    raw_events, stream_name, "adaptive"
                )
                
                stream_data[stream_name] = processed_events
                processing_reports[stream_name] = report
                
                # Add events to synchronizer buffer
                for event in processed_events:
                    self.synchronizer.add_event_to_buffer(stream_name, event)
                
            else:
                logger.warning(f"Stream file not found: {file_path}")
                processing_reports[stream_name] = {"error": "file_not_found"}
        
        # Step 2: Synchronize events across time windows
        logger.info("Synchronizing events across time windows...")
        
        # Determine time range for synchronization
        all_timestamps = []
        for events in stream_data.values():
            for event in events:
                try:
                    ts = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                    all_timestamps.append(ts)
                except:
                    continue
        
        if not all_timestamps:
            logger.error("No valid timestamps found in data streams")
            return {"error": "no_valid_timestamps"}
        
        start_time = min(all_timestamps)
        end_time = max(all_timestamps)
        
        # Synchronize in 1-minute intervals
        synchronized_events = []
        current_time = start_time
        
        while current_time <= end_time:
            sync_events = self.synchronizer.synchronize_by_time_window(current_time)
            synchronized_events.extend(sync_events)
            current_time += timedelta(minutes=1)
        
        logger.info(f"Generated {len(synchronized_events)} synchronized events")
        
        # Step 3: Cross-stream correlation
        logger.info("Performing cross-stream correlation...")
        correlations = self.correlator.correlate_checkout_activity(synchronized_events)
        
        # Step 4: Generate comprehensive results
        integration_results = {
            "processing_summary": {
                "streams_processed": len(stream_data),
                "total_synchronized_events": len(synchronized_events),
                "correlations_found": len(correlations),
                "processing_reports": processing_reports
            },
            "synchronized_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "station_id": event.station_id,
                    "correlation_confidence": event.correlation_confidence,
                    "data_quality_score": event.data_quality_score,
                    "anomaly_flags": event.anomaly_flags,
                    "missing_streams": event.missing_streams,
                    "event_sources": list(event.correlated_events.keys())
                }
                for event in synchronized_events
            ],
            "correlations": correlations,
            "data_quality_metrics": self._calculate_quality_metrics(synchronized_events),
            "anomaly_summary": self._summarize_anomalies(synchronized_events)
        }
        
        logger.info("Data integration pipeline completed successfully")
        return integration_results
    
    def _load_jsonl_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load events from JSONL file"""
        events = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            events.append(event)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
        
        return events
    
    def _calculate_quality_metrics(self, events: List[SynchronizedEvent]) -> Dict[str, float]:
        """Calculate overall data quality metrics"""
        if not events:
            return {}
        
        quality_scores = [e.data_quality_score for e in events]
        correlation_scores = [e.correlation_confidence for e in events]
        
        return {
            "average_data_quality": statistics.mean(quality_scores),
            "min_data_quality": min(quality_scores),
            "max_data_quality": max(quality_scores),
            "average_correlation_confidence": statistics.mean(correlation_scores),
            "events_with_high_quality": len([s for s in quality_scores if s >= 0.8]) / len(quality_scores),
            "events_with_anomalies": len([e for e in events if e.anomaly_flags]) / len(events)
        }
    
    def _summarize_anomalies(self, events: List[SynchronizedEvent]) -> Dict[str, int]:
        """Summarize anomaly patterns across all events"""
        anomaly_counts = defaultdict(int)
        
        for event in events:
            for flag in event.anomaly_flags:
                anomaly_counts[flag] += 1
        
        return dict(anomaly_counts)


# Main execution function for testing
async def main():
    """Test the data integration pipeline"""
    
    pipeline = DataIntegrationPipeline("data/input")
    results = await pipeline.process_all_streams()
    
    # Print summary
    print("\n" + "="*60)
    print("DATA INTEGRATION PIPELINE RESULTS")
    print("="*60)
    
    summary = results.get("processing_summary", {})
    print(f"Streams Processed: {summary.get('streams_processed', 0)}")
    print(f"Synchronized Events: {summary.get('total_synchronized_events', 0)}")
    print(f"Correlations Found: {summary.get('correlations_found', 0)}")
    
    if "data_quality_metrics" in results:
        metrics = results["data_quality_metrics"]
        print(f"\nData Quality Metrics:")
        print(f"  Average Quality Score: {metrics.get('average_data_quality', 0):.2f}")
        print(f"  Average Correlation Confidence: {metrics.get('average_correlation_confidence', 0):.2f}")
        print(f"  High Quality Events: {metrics.get('events_with_high_quality', 0):.1%}")
    
    if "anomaly_summary" in results:
        anomalies = results["anomaly_summary"]
        if anomalies:
            print(f"\nAnomalies Detected:")
            for anomaly_type, count in anomalies.items():
                print(f"  {anomaly_type}: {count}")
    
    # Save results
    output_file = Path("data/output/integration_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())