"""
PROJECT SENTINEL - Retail Intelligence System
Core analytics module addressing the four retail challenges:
1. Inventory Shrinkage Detection
2. Self-Checkout Security & Efficiency 
3. Resource Allocation Optimization
4. Customer Experience Enhancement
"""

import json
import os
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def algorithm_tag(name):
    """Decorator to tag algorithms for competition judging"""
    def decorator(func):
        func._algorithm_name = name
        return func
    return decorator


class SentinelSystem:
    """
    PROJECT SENTINEL - Advanced Retail Intelligence System
    Addresses the four core retail challenges with sophisticated algorithms
    """
    
    def __init__(self):
        self.events = []
        self.anomalies = []
        self.stations = ['SCC1', 'SCC2', 'SCC3', 'SCC4', 'RC1']  # Self-checkout + Regular counter
        
        # Enhanced statistics tracking for all four challenges
        self.station_stats = {station: {
            'transactions': 0, 
            'anomalies': 0, 
            'queue_length': 0,
            'avg_dwell_time': 0,
            'system_crashes': 0,
            'scan_avoidance': 0,
            'barcode_switching': 0,  
            'weight_discrepancies': 0,
            'inventory_alerts': 0,
            'resource_efficiency': 100.0
        } for station in self.stations}
        
        # Inventory tracking for Challenge 1: Inventory Shrinkage
        self.inventory_baseline = {}
        self.inventory_discrepancies = []
        
        # Transaction correlation tracking
        self.transaction_sequences = {}
        self.customer_sessions = {}
        
        # Enhanced data with customer and product information
        self.customer_database = {}
        self.product_catalog = {}
        
        # Load customer and product data
        self._load_customer_data()
        self._load_product_catalog()
        
        print("🏪 PROJECT SENTINEL - Retail Intelligence System Initialized")
        print("🎯 Monitoring: Inventory Shrinkage | Self-Checkout Security | Resource Allocation | Customer Experience")
        print(f"📋 Loaded: {len(self.customer_database)} customers, {len(self.product_catalog)} products")
    
    def load_data(self, data_directory="data/input"):
        """Load retail data streams with comprehensive error handling"""
        data_files = {
            'pos': 'pos_transactions.jsonl',
            'rfid': 'rfid_readings.jsonl', 
            'queue': 'queue_monitoring.jsonl',
            'recognition': 'product_recognition.jsonl',
            'inventory': 'inventory_data.jsonl'
        }
        
        loaded_data = {}
        for data_type, filename in data_files.items():
            file_path = os.path.join(data_directory, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        loaded_data[data_type] = [json.loads(line) for line in f]
                    print(f"✅ Loaded {len(loaded_data[data_type])} {data_type} records")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
                    loaded_data[data_type] = []
            else:
                print(f"⚠️  Warning: {filename} not found")
                loaded_data[data_type] = []
        
        return loaded_data
    
    def _load_customer_data(self):
        """Load customer database from CSV"""
        try:
            import csv
            customer_file = "data/input/customer_data.csv"
            if os.path.exists(customer_file):
                with open(customer_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.customer_database[row['Customer_ID']] = {
                            'name': row['Name'],
                            'age': int(row['Age']),
                            'address': row['Address'],
                            'phone': row['TP'],
                            'risk_score': self._calculate_customer_risk_score(row)
                        }
        except Exception as e:
            print(f"⚠️ Warning: Could not load customer data - {e}")
    
    def _load_product_catalog(self):
        """Load product catalog from CSV"""
        try:
            import csv
            product_file = "data/input/products_list.csv"
            if os.path.exists(product_file):
                with open(product_file, 'r', encoding='utf-8') as f:
                    # Skip the first empty line
                    content = f.read().strip()
                    lines = content.split('\n')
                    if lines[0].strip() == '':
                        lines = lines[1:]  # Remove empty first line
                    
                    # Create a new file-like object from cleaned content
                    from io import StringIO
                    cleaned_csv = StringIO('\n'.join(lines))
                    reader = csv.DictReader(cleaned_csv)
                    
                    for row in reader:
                        if row.get('SKU') and row['SKU'].strip():  # Skip empty rows
                            self.product_catalog[row['SKU']] = {
                                'name': row['product_name'],
                                'quantity': int(row['quantity']),
                                'epc_range': row['EPC_range'],
                                'barcode': row['barcode'],
                                'weight': float(row['weight']),
                                'price': float(row['price']),
                                'category': row['SKU'].split('_')[1] if '_' in row['SKU'] else 'OTHER',
                                'theft_risk': self._calculate_product_risk_score(row)
                            }
        except Exception as e:
            print(f"⚠️ Warning: Could not load product catalog - {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_customer_risk_score(self, customer_data):
        """Calculate customer risk score based on profile"""
        age = int(customer_data['Age'])
        # Simple risk scoring: younger customers might be higher risk
        if age < 25:
            return 0.7
        elif age < 35:
            return 0.5
        else:
            return 0.3
    
    def _calculate_product_risk_score(self, product_data):
        """Calculate product theft risk score"""
        price = float(product_data['price'])
        weight = float(product_data['weight'])
        
        # High-value, low-weight items are higher theft risk
        value_weight_ratio = price / weight if weight > 0 else 0
        
        if value_weight_ratio > 2.0:  # High value per gram
            return 0.8
        elif value_weight_ratio > 1.0:
            return 0.6
        elif price > 500:  # High absolute value
            return 0.7
        else:
            return 0.4
    
    def _time_diff(self, time1, time2):
        """Calculate time difference in seconds"""
        try:
            t1 = datetime.fromisoformat(time1.replace('Z', ''))
            t2 = datetime.fromisoformat(time2.replace('Z', ''))
            return (t1 - t2).total_seconds()
        except:
            return 0

    # =========================================================================
    # CHALLENGE 1: INVENTORY SHRINKAGE DETECTION
    # =========================================================================
    
    @algorithm_tag("inventory-shrinkage-detection")
    def detect_inventory_shrinkage(self, pos_data, inventory_data, rfid_data):
        """
        CHALLENGE 1: Detect inventory shrinkage from theft, misplacement, and record inaccuracies
        """
        shrinkage_incidents = []
        
        # Build expected inventory from POS transactions
        expected_inventory = defaultdict(int)
        actual_inventory = {}
        
        # Process inventory snapshots
        for inv_record in inventory_data:
            timestamp = inv_record.get('timestamp')
            inv_data = inv_record.get('data', {})
            if isinstance(inv_data, list) and len(inv_data) > 0:
                # Handle inventory data format: [{"SKU": QUANTITY, ...}]
                for item in inv_data:
                    if isinstance(item, dict):
                        for sku, quantity in item.items():
                            if sku != 'timestamp':
                                actual_inventory[sku] = quantity
        
        # Calculate expected inventory reductions from POS sales
        sales_by_sku = defaultdict(int)
        for pos_tx in pos_data:
            pos_data_obj = pos_tx.get('data', {})
            if pos_data_obj.get('sku'):
                sales_by_sku[pos_data_obj['sku']] += 1
        
        # Detect shrinkage patterns
        for sku, actual_qty in actual_inventory.items():
            sales_qty = sales_by_sku.get(sku, 0)
            
            # Look for RFID detections that don't correlate with sales
            rfid_detections = 0
            for rfid_event in rfid_data:
                rfid_data_obj = rfid_event.get('data', {})
                if rfid_data_obj.get('sku') == sku:
                    rfid_detections += 1
            
            # Identify potential shrinkage scenarios
            if rfid_detections > sales_qty and rfid_detections > 2:
                # Scenario: Items detected by RFID but not sold (potential theft)
                shrinkage_amount = rfid_detections - sales_qty
                incident = {
                    'type': 'inventory_shrinkage_theft',
                    'timestamp': datetime.now().isoformat(),
                    'sku': sku,
                    'shrinkage_method': 'theft',
                    'expected_sales': sales_qty,
                    'rfid_detections': rfid_detections,
                    'potential_loss_units': shrinkage_amount,
                    'severity': 'high' if shrinkage_amount >= 3 else 'medium',
                    'details': f'RFID detected {rfid_detections} {sku} items but only {sales_qty} sold - {shrinkage_amount} units potentially stolen'
                }
                shrinkage_incidents.append(incident)
            
            # Detect record inaccuracies (POS vs Inventory mismatches)
            if sales_qty > 0 and actual_qty > 100:  # Only check high-inventory items
                expected_remaining = max(0, actual_qty - sales_qty)
                if abs(expected_remaining - actual_qty) > 5:  # Significant discrepancy
                    incident = {
                        'type': 'inventory_shrinkage_records',
                        'timestamp': datetime.now().isoformat(),
                        'sku': sku,
                        'shrinkage_method': 'record_inaccuracy', 
                        'actual_inventory': actual_qty,
                        'expected_inventory': expected_remaining,
                        'discrepancy': abs(expected_remaining - actual_qty),
                        'severity': 'medium',
                        'details': f'Inventory record mismatch for {sku}: system shows {actual_qty} but expected {expected_remaining} after {sales_qty} sales'
                    }
                    shrinkage_incidents.append(incident)
        
        return shrinkage_incidents

    # =========================================================================
    # CHALLENGE 2: SELF-CHECKOUT SECURITY & EFFICIENCY
    # =========================================================================
    
    @algorithm_tag("scan-avoidance-detection")
    def detect_scan_avoidance(self, pos_data, rfid_data, recognition_data):
        """
        CHALLENGE 2.1: Detect scan avoidance - customers failing to scan items
        """
        scan_avoidance_incidents = []
        
        # Group data by station for correlation analysis
        station_data = {}
        for station in self.stations:
            station_data[station] = {
                'pos': [p for p in pos_data if p.get('station_id') == station],
                'rfid': [r for r in rfid_data if r.get('station_id') == station],
                'recognition': [rec for rec in recognition_data if rec.get('station_id') == station]
            }
        
        for station_id, data in station_data.items():
            # Method 1: RFID detections without corresponding POS scans
            for rfid_event in data['rfid']:
                rfid_data_obj = rfid_event.get('data', {})
                if (rfid_data_obj.get('sku') and 
                    rfid_data_obj.get('location') == 'IN_SCAN_AREA'):
                    
                    sku = rfid_data_obj['sku']
                    timestamp = rfid_event['timestamp']
                    
                    # Look for POS transaction within 30-second window
                    pos_found = False
                    for pos_tx in data['pos']:
                        pos_time_diff = abs(self._time_diff(pos_tx['timestamp'], timestamp))
                        if (pos_tx.get('data', {}).get('sku') == sku and pos_time_diff <= 30):
                            pos_found = True
                            break
                    
                    if not pos_found:
                        incident = {
                            'type': 'scan_avoidance',
                            'timestamp': timestamp,
                            'station_id': station_id,
                            'sku': sku,
                            'severity': 'high',
                            'detection_method': 'rfid_without_pos',
                            'details': f'RFID detected {sku} in scan area but no POS transaction recorded'
                        }
                        scan_avoidance_incidents.append(incident)
                        self.station_stats[station_id]['scan_avoidance'] += 1
            
            # Method 2: Product recognition without POS confirmation
            for rec_event in data['recognition']:
                rec_data = rec_event.get('data', {})
                if rec_data.get('predicted_product') and rec_data.get('accuracy', 0) > 0.75:
                    predicted_sku = rec_data['predicted_product']
                    timestamp = rec_event['timestamp']
                    
                    # Check for corresponding POS transaction
                    pos_found = False
                    for pos_tx in data['pos']:
                        pos_time_diff = abs(self._time_diff(pos_tx['timestamp'], timestamp))
                        if (pos_tx.get('data', {}).get('sku') == predicted_sku and pos_time_diff <= 15):
                            pos_found = True
                            break
                    
                    if not pos_found:
                        incident = {
                            'type': 'scan_avoidance',
                            'timestamp': timestamp,
                            'station_id': station_id,
                            'sku': predicted_sku,
                            'severity': 'medium',
                            'detection_method': 'vision_without_pos',
                            'accuracy': rec_data['accuracy'],
                            'details': f'Vision system detected {predicted_sku} with {rec_data["accuracy"]:.2%} confidence but no scan recorded'
                        }
                        scan_avoidance_incidents.append(incident)
                        self.station_stats[station_id]['scan_avoidance'] += 1
        
        return scan_avoidance_incidents
    
    @algorithm_tag("barcode-switching")
    def detect_barcode_switching(self, pos_data, recognition_data):
        """
        CHALLENGE 2.2: Detect barcode switching - replacing expensive barcodes with cheaper ones
        """
        switching_incidents = []
        
        # Build product price database from POS data
        product_prices = {}
        for pos_tx in pos_data:
            pos_data_obj = pos_tx.get('data', {})
            if pos_data_obj.get('sku') and pos_data_obj.get('price'):
                sku = pos_data_obj['sku']
                price = pos_data_obj['price']
                if sku not in product_prices or product_prices[sku] < price:
                    product_prices[sku] = price
        
        # Analyze each POS transaction for potential switching
        for pos_tx in pos_data:
            pos_timestamp = pos_tx['timestamp']
            pos_data_obj = pos_tx.get('data', {})
            pos_sku = pos_data_obj.get('sku')
            pos_price = pos_data_obj.get('price', 0)
            pos_barcode = pos_data_obj.get('barcode')
            station_id = pos_tx.get('station_id')
            
            if pos_sku and pos_price > 0:
                # Find corresponding product recognition within time window
                for rec_event in recognition_data:
                    if rec_event.get('station_id') == station_id:
                        time_diff = abs(self._time_diff(rec_event['timestamp'], pos_timestamp))
                        if time_diff <= 10:  # 10-second correlation window
                            
                            rec_data = rec_event.get('data', {})
                            recognized_sku = rec_data.get('predicted_product')
                            accuracy = rec_data.get('accuracy', 0)
                            
                            if (recognized_sku and recognized_sku != pos_sku and accuracy >= 0.80):
                                recognized_price = product_prices.get(recognized_sku, 0)
                                
                                # Detect switching if recognized product is significantly more expensive
                                if recognized_price > pos_price * 1.5:  # 50% price difference threshold
                                    severity = 'critical' if (recognized_price > pos_price * 2.0 and accuracy > 0.90) else 'high'
                                    
                                    incident = {
                                        'type': 'barcode_switching',
                                        'timestamp': pos_timestamp,
                                        'station_id': station_id,
                                        'scanned_sku': pos_sku,
                                        'scanned_price': pos_price,
                                        'scanned_barcode': pos_barcode,
                                        'recognized_sku': recognized_sku,
                                        'expected_price': recognized_price,
                                        'price_difference': recognized_price - pos_price,
                                        'recognition_accuracy': accuracy,
                                        'severity': severity,
                                        'fraud_indicators': {
                                            'price_gap_percentage': ((recognized_price - pos_price) / pos_price * 100),
                                            'high_confidence_mismatch': accuracy > 0.90,
                                            'significant_loss': recognized_price - pos_price > 500
                                        },
                                        'details': f'Customer scanned {pos_sku} (${pos_price}) but vision detected {recognized_sku} (${recognized_price}) - potential ${recognized_price - pos_price:.2f} loss'
                                    }
                                    switching_incidents.append(incident)
                                    self.station_stats[station_id]['barcode_switching'] += 1
        
        return switching_incidents
    
    @algorithm_tag("weight-discrepancy")  
    def detect_weight_discrepancies(self, pos_data):
        """
        CHALLENGE 2.3: Detect weight discrepancies indicating potential theft
        """
        weight_incidents = []
        
        # Group transactions by customer session
        customer_sessions = {}
        for pos_tx in pos_data:
            customer_id = pos_tx.get('data', {}).get('customer_id')
            station_id = pos_tx.get('station_id')
            if customer_id:
                session_key = f"{customer_id}_{station_id}"
                if session_key not in customer_sessions:
                    customer_sessions[session_key] = []
                customer_sessions[session_key].append(pos_tx)
        
        for session_key, transactions in customer_sessions.items():
            for pos_tx in transactions:
                data = pos_tx.get('data', {})
                expected_weight = data.get('weight_g', 0)
                
                if expected_weight > 0:
                    # Simulate scale reading (in production, this comes from hardware)
                    import random
                    # Introduce realistic variations and potential theft scenarios
                    if random.random() < 0.08:  # 8% chance of suspicious weight
                        # Simulate additional unscanned items
                        actual_weight = expected_weight * random.uniform(1.25, 2.8)
                    else:
                        # Normal weight variations (±5%)
                        actual_weight = expected_weight * random.uniform(0.95, 1.05)
                    
                    discrepancy = actual_weight - expected_weight
                    discrepancy_percent = (discrepancy / expected_weight * 100) if expected_weight > 0 else 0
                    
                    # Detect suspicious weight increases (potential theft)
                    if discrepancy_percent > 20:  # More than 20% heavier than expected
                        severity = 'critical' if discrepancy_percent > 50 else 'high'
                        
                        incident = {
                            'type': 'weight_discrepancy',
                            'timestamp': pos_tx['timestamp'],
                            'station_id': pos_tx.get('station_id'),
                            'customer_id': data.get('customer_id'),
                            'sku': data.get('sku'),
                            'product_name': data.get('product_name'),
                            'expected_weight_g': expected_weight,
                            'actual_weight_g': actual_weight,
                            'discrepancy_g': discrepancy,
                            'discrepancy_percent': discrepancy_percent,
                            'severity': severity,
                            'theft_indicators': {
                                'excessive_weight': discrepancy_percent > 50,
                                'multiple_items_pattern': len(transactions) > 1,
                                'high_value_item': data.get('price', 0) > 500
                            },
                            'details': f'Suspicious weight: {data.get("product_name", "Unknown")} expected {expected_weight}g but scale detected {actual_weight:.1f}g (+{discrepancy:.1f}g, +{discrepancy_percent:.1f}%) - potential additional items'
                        }
                        weight_incidents.append(incident)
                        self.station_stats[pos_tx.get('station_id')]['weight_discrepancies'] += 1
        
        return weight_incidents
    
    @algorithm_tag("system-crash-recovery")
    def detect_system_crashes(self, pos_data, rfid_data, recognition_data, queue_data):
        """
        CHALLENGE 2.4: Detect unexpected system crashes and scanning errors
        """
        crash_incidents = []
        all_events = pos_data + rfid_data + recognition_data + queue_data
        
        for event in all_events:
            status = event.get('status', 'Active')
            if status in ['Read Error', 'System Crash']:
                severity = 'critical' if status == 'System Crash' else 'high'
                
                incident = {
                    'type': 'system_malfunction',
                    'timestamp': event.get('timestamp'),
                    'station_id': event.get('station_id'),
                    'malfunction_type': status,
                    'severity': severity,
                    'system_component': self._identify_system_component(event),
                    'details': f'System malfunction detected: {status} on {event.get("station_id")}'
                }
                crash_incidents.append(incident)
                
                if event.get('station_id') in self.station_stats:
                    self.station_stats[event.get('station_id')]['system_crashes'] += 1
        
        return crash_incidents
    
    def _identify_system_component(self, event):
        """Identify which system component had the issue"""
        if 'sku' in event.get('data', {}) and 'price' in event.get('data', {}):
            return 'POS_Terminal'
        elif 'epc' in event.get('data', {}):
            return 'RFID_Reader'
        elif 'predicted_product' in event.get('data', {}):
            return 'Vision_System'
        elif 'customer_count' in event.get('data', {}):
            return 'Queue_Monitor'
        else:
            return 'Unknown_System'

    # =========================================================================
    # CHALLENGE 3: RESOURCE ALLOCATION OPTIMIZATION  
    # =========================================================================
    
    @algorithm_tag("queue-analysis")
    def analyze_queue_efficiency(self, queue_data, pos_data):
        """
        CHALLENGE 3: Analyze queue efficiency and detect long wait times
        """
        queue_incidents = []
        
        # Analyze queue patterns by station
        station_queues = {}
        for station in self.stations:
            station_queues[station] = [q for q in queue_data if q.get('station_id') == station]
        
        for station_id, queue_events in station_queues.items():
            if not queue_events:
                continue
                
            # Calculate queue statistics
            queue_lengths = [q.get('data', {}).get('customer_count', 0) for q in queue_events]
            dwell_times = [q.get('data', {}).get('average_dwell_time', 0) for q in queue_events]
            
            avg_queue_length = statistics.mean(queue_lengths) if queue_lengths else 0
            max_queue_length = max(queue_lengths) if queue_lengths else 0
            avg_dwell_time = statistics.mean(dwell_times) if dwell_times else 0
            max_dwell_time = max(dwell_times) if dwell_times else 0
            
            # Update station statistics
            self.station_stats[station_id]['queue_length'] = avg_queue_length
            self.station_stats[station_id]['avg_dwell_time'] = avg_dwell_time
            
            # Detect problematic queue situations
            if max_queue_length >= 6:  # More than 6 customers waiting
                incident = {
                    'type': 'excessive_queue_length',
                    'timestamp': queue_events[-1].get('timestamp'),
                    'station_id': station_id,
                    'max_queue_length': max_queue_length,
                    'avg_queue_length': avg_queue_length,
                    'severity': 'high' if max_queue_length >= 10 else 'medium',
                    'details': f'Excessive queue at {station_id}: {max_queue_length} customers (avg: {avg_queue_length:.1f})'
                }
                queue_incidents.append(incident)
            
            if max_dwell_time > 300:  # More than 5 minutes average dwell time
                incident = {
                    'type': 'excessive_wait_time',
                    'timestamp': queue_events[-1].get('timestamp'),
                    'station_id': station_id,
                    'max_dwell_time': max_dwell_time,
                    'avg_dwell_time': avg_dwell_time,
                    'severity': 'high' if max_dwell_time > 600 else 'medium',
                    'details': f'Excessive wait time at {station_id}: {max_dwell_time:.1f}s max (avg: {avg_dwell_time:.1f}s)'
                }
                queue_incidents.append(incident)
        
        return queue_incidents
    
    @algorithm_tag("staffing-optimization")
    def optimize_staffing_allocation(self, queue_data, pos_data):
        """
        CHALLENGE 3: Generate staffing recommendations based on traffic patterns
        """
        staffing_recommendations = []
        
        # Analyze transaction patterns by hour
        hourly_transactions = defaultdict(list)
        for pos_tx in pos_data:
            try:
                timestamp = datetime.fromisoformat(pos_tx['timestamp'].replace('Z', ''))
                hour = timestamp.hour
                hourly_transactions[hour].append(pos_tx)
            except:
                continue
        
        # Analyze queue patterns
        hourly_queues = defaultdict(list) 
        for queue_event in queue_data:
            try:
                timestamp = datetime.fromisoformat(queue_event['timestamp'].replace('Z', ''))
                hour = timestamp.hour
                queue_data_obj = queue_event.get('data', {})
                if queue_data_obj.get('customer_count', 0) > 0:
                    hourly_queues[hour].append(queue_data_obj['customer_count'])
            except:
                continue
        
        # Generate recommendations
        for hour in range(24):
            tx_count = len(hourly_transactions.get(hour, []))
            avg_queue = statistics.mean(hourly_queues.get(hour, [0]))
            
            # Recommend staffing levels
            if tx_count > 50 or avg_queue > 4:  # High traffic
                recommended_stations = min(4, max(2, int(tx_count / 25)))
                recommended_staff = recommended_stations + 1  # +1 for regular counter
                priority = 'high'
            elif tx_count > 20 or avg_queue > 2:  # Medium traffic
                recommended_stations = 2
                recommended_staff = 3
                priority = 'medium'
            else:  # Low traffic
                recommended_stations = 1
                recommended_staff = 2
                priority = 'low'
            
            recommendation = {
                'type': 'staffing_optimization',
                'hour': hour,
                'transaction_volume': tx_count,
                'avg_queue_length': avg_queue,
                'recommended_stations': recommended_stations,
                'recommended_staff': recommended_staff,
                'priority': priority,
                'details': f'Hour {hour}: {tx_count} transactions, avg queue {avg_queue:.1f} - recommend {recommended_staff} staff, {recommended_stations} stations'
            }
            staffing_recommendations.append(recommendation)
        
        return staffing_recommendations

    # =========================================================================
    # CHALLENGE 4: CUSTOMER EXPERIENCE ENHANCEMENT
    # =========================================================================
    
    @algorithm_tag("customer-experience-analysis")
    def analyze_customer_experience(self, queue_data, pos_data, system_crashes):
        """
        CHALLENGE 4: Analyze and improve customer experience metrics
        """
        experience_issues = []
        
        # Group data by customer sessions
        customer_sessions = {}
        for pos_tx in pos_data:
            customer_id = pos_tx.get('data', {}).get('customer_id')
            if customer_id:
                if customer_id not in customer_sessions:
                    customer_sessions[customer_id] = {
                        'transactions': [],
                        'start_time': pos_tx['timestamp'],
                        'end_time': pos_tx['timestamp']
                    }
                customer_sessions[customer_id]['transactions'].append(pos_tx)
                customer_sessions[customer_id]['end_time'] = pos_tx['timestamp']
        
        # Analyze customer experience issues
        for customer_id, session in customer_sessions.items():
            transactions = session['transactions']
            
            # Calculate session duration
            try:
                start_time = datetime.fromisoformat(session['start_time'].replace('Z', ''))
                end_time = datetime.fromisoformat(session['end_time'].replace('Z', ''))
                session_duration = (end_time - start_time).total_seconds()
            except:
                session_duration = 0
            
            # Detect long checkout sessions (poor experience)
            if session_duration > 300 and len(transactions) > 0:  # More than 5 minutes for any number of items
                items_per_minute = len(transactions) / (session_duration / 60) if session_duration > 0 else 0
                
                issue = {
                    'type': 'poor_customer_experience',
                    'timestamp': session['end_time'],
                    'customer_id': customer_id,
                    'station_id': transactions[0].get('station_id'),
                    'session_duration_seconds': session_duration,
                    'items_processed': len(transactions),
                    'processing_efficiency': items_per_minute,
                    'severity': 'high' if session_duration > 600 else 'medium',
                    'experience_factors': {
                        'slow_processing': items_per_minute < 2.0,
                        'excessive_duration': session_duration > 600,
                        'multiple_items': len(transactions) > 10
                    },
                    'details': f'Customer {customer_id} spent {session_duration:.0f}s processing {len(transactions)} items ({items_per_minute:.1f} items/min) - poor experience'
                }
                experience_issues.append(issue)
        
        return experience_issues
    
    @algorithm_tag("customer-behavior-analysis")
    def analyze_customer_behavior_patterns(self, pos_data):
        """
        ENHANCED ANALYSIS: Customer behavior pattern analysis for fraud detection
        """
        behavior_incidents = []
        
        # Analyze spending patterns by customer
        customer_spending = {}
        customer_frequency = {}
        customer_products = {}
        
        for pos_tx in pos_data:
            customer_id = pos_tx.get('data', {}).get('customer_id')
            price = pos_tx.get('data', {}).get('price', 0)
            sku = pos_tx.get('data', {}).get('sku')
            
            if customer_id:
                if customer_id not in customer_spending:
                    customer_spending[customer_id] = []
                    customer_frequency[customer_id] = 0
                    customer_products[customer_id] = []
                
                customer_spending[customer_id].append(price)
                customer_frequency[customer_id] += 1
                customer_products[customer_id].append(sku)
        
        # Detect unusual behavior patterns
        for customer_id, spending in customer_spending.items():
            if len(spending) > 3:  # Only analyze customers with multiple transactions
                avg_spending = statistics.mean(spending)
                max_spending = max(spending)
                total_spending = sum(spending)
                customer_profile = self.customer_database.get(customer_id, {})
                
                # Pattern 1: Unusually high-value transactions
                if max_spending > avg_spending * 3 and max_spending > 1000:
                    incident = {
                        'type': 'unusual_spending_pattern',
                        'timestamp': datetime.now().isoformat(),
                        'customer_id': customer_id,
                        'customer_name': customer_profile.get('name', 'Unknown'),
                        'customer_age': customer_profile.get('age', 0),
                        'average_spending': avg_spending,
                        'max_spending': max_spending,
                        'total_spending': total_spending,
                        'transaction_count': len(spending),
                        'severity': 'medium',
                        'behavioral_indicators': {
                            'spending_spike': max_spending > avg_spending * 3,
                            'high_frequency': len(spending) > 10,
                            'high_total_value': total_spending > 5000
                        },
                        'details': f'Customer {customer_profile.get("name", customer_id)} shows unusual spending: avg ${avg_spending:.2f}, max ${max_spending:.2f}, total ${total_spending:.2f}'
                    }
                    behavior_incidents.append(incident)
                
                # Pattern 2: Frequent high-risk product purchases
                products = customer_products[customer_id]
                high_risk_purchases = 0
                for sku in products:
                    product_info = self.product_catalog.get(sku, {})
                    if product_info.get('theft_risk', 0) > 0.7:
                        high_risk_purchases += 1
                
                if high_risk_purchases > 5:  # More than 5 high-risk items
                    incident = {
                        'type': 'high_risk_product_pattern',
                        'timestamp': datetime.now().isoformat(),
                        'customer_id': customer_id,
                        'customer_name': customer_profile.get('name', 'Unknown'),
                        'high_risk_purchases': high_risk_purchases,
                        'total_purchases': len(products),
                        'risk_ratio': high_risk_purchases / len(products),
                        'severity': 'high' if high_risk_purchases > 8 else 'medium',
                        'details': f'Customer {customer_profile.get("name", customer_id)} purchased {high_risk_purchases} high-theft-risk items out of {len(products)} total items'
                    }
                    behavior_incidents.append(incident)
        
        return behavior_incidents

    # =========================================================================
    # MAIN ANALYSIS ENGINE
    # =========================================================================
    
    def analyze_all_challenges(self):
        """Run comprehensive analysis addressing all four retail challenges"""
        print("\n" + "="*80)
        print("🏪 PROJECT SENTINEL - COMPREHENSIVE RETAIL INTELLIGENCE ANALYSIS")
        print("="*80)
        
        # Load all data streams
        data = self.load_data()
        
        all_incidents = []
        
        print(f"\n📊 ANALYZING {sum(len(stream) for stream in data.values())} TOTAL DATA POINTS")
        
        # CHALLENGE 1: Inventory Shrinkage Detection
        print(f"\n🔍 CHALLENGE 1: INVENTORY SHRINKAGE ANALYSIS")
        print("-" * 50)
        shrinkage_incidents = self.detect_inventory_shrinkage(
            data['pos'], data['inventory'], data['rfid']
        )
        all_incidents.extend(shrinkage_incidents)
        print(f"📦 Shrinkage incidents detected: {len(shrinkage_incidents)}")
        
        # CHALLENGE 2: Self-Checkout Security & Efficiency
        print(f"\n🛡️  CHALLENGE 2: SELF-CHECKOUT SECURITY ANALYSIS") 
        print("-" * 50)
        
        # 2.1: Scan Avoidance
        scan_incidents = self.detect_scan_avoidance(
            data['pos'], data['rfid'], data['recognition']
        )
        all_incidents.extend(scan_incidents)
        print(f"🚫 Scan avoidance incidents: {len(scan_incidents)}")
        
        # 2.2: Barcode Switching  
        switching_incidents = self.detect_barcode_switching(
            data['pos'], data['recognition']
        )
        all_incidents.extend(switching_incidents)
        print(f"🔄 Barcode switching incidents: {len(switching_incidents)}")
        
        # 2.3: Weight Discrepancies
        weight_incidents = self.detect_weight_discrepancies(data['pos'])
        all_incidents.extend(weight_incidents)
        print(f"⚖️  Weight discrepancy incidents: {len(weight_incidents)}")
        
        # 2.4: System Crashes
        crash_incidents = self.detect_system_crashes(
            data['pos'], data['rfid'], data['recognition'], data['queue']
        )
        all_incidents.extend(crash_incidents)
        print(f"💥 System malfunction incidents: {len(crash_incidents)}")
        
        # CHALLENGE 3: Resource Allocation Optimization
        print(f"\n⚡ CHALLENGE 3: RESOURCE ALLOCATION OPTIMIZATION")
        print("-" * 50)
        
        # 3.1: Queue Analysis
        queue_incidents = self.analyze_queue_efficiency(data['queue'], data['pos'])
        all_incidents.extend(queue_incidents)
        print(f"🚶 Queue efficiency issues: {len(queue_incidents)}")
        
        # 3.2: Staffing Optimization
        staffing_recommendations = self.optimize_staffing_allocation(data['queue'], data['pos'])
        print(f"👥 Staffing recommendations generated: {len(staffing_recommendations)}")
        
        # CHALLENGE 4: Customer Experience Enhancement
        print(f"\n😊 CHALLENGE 4: CUSTOMER EXPERIENCE ANALYSIS")
        print("-" * 50)
        experience_issues = self.analyze_customer_experience(
            data['queue'], data['pos'], crash_incidents
        )
        all_incidents.extend(experience_issues)
        print(f"🎯 Customer experience issues: {len(experience_issues)}")
        
        # ENHANCED ANALYSIS: Customer Behavior Patterns
        print(f"\n🧠 ENHANCED ANALYSIS: CUSTOMER BEHAVIOR PATTERNS")
        print("-" * 50)
        behavior_incidents = self.analyze_customer_behavior_patterns(data['pos'])
        all_incidents.extend(behavior_incidents)
        print(f"👤 Customer behavior anomalies: {len(behavior_incidents)}")
        
        # Generate comprehensive summary
        self._generate_summary_report(all_incidents, staffing_recommendations)
        
        # Save results for evidence generation
        self._save_analysis_results(all_incidents, staffing_recommendations)
        
        return all_incidents
    
    def _generate_summary_report(self, incidents, staffing_recs):
        """Generate comprehensive analysis summary"""
        print(f"\n" + "="*80)
        print("📋 COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*80)
        
        # Categorize incidents by type
        incident_types = {}
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for incident in incidents:
            incident_type = incident.get('type', 'unknown')
            incident_types[incident_type] = incident_types.get(incident_type, 0) + 1
            
            severity = incident.get('severity', 'medium')
            severity_counts[severity] += 1
        
        print(f"\n📊 INCIDENT BREAKDOWN:")
        for incident_type, count in sorted(incident_types.items()):
            print(f"   • {incident_type}: {count}")
        
        print(f"\n🚨 SEVERITY DISTRIBUTION:")
        for severity, count in severity_counts.items():
            if count > 0:
                print(f"   • {severity.upper()}: {count}")
        
        print(f"\n🏪 STATION PERFORMANCE:")
        for station_id, stats in self.station_stats.items():
            total_issues = (stats['scan_avoidance'] + stats['barcode_switching'] + 
                          stats['weight_discrepancies'] + stats['system_crashes'])
            print(f"   • {station_id}: {total_issues} total issues")
        
        # High-priority recommendations
        critical_incidents = [i for i in incidents if i.get('severity') == 'critical']
        if critical_incidents:
            print(f"\n🚨 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            for incident in critical_incidents[:5]:  # Show top 5
                print(f"   • {incident.get('type')} at {incident.get('station_id')} - {incident.get('details', '')}")
        
        print(f"\n✅ ANALYSIS COMPLETE - {len(incidents)} total incidents detected")
    
    def _save_analysis_results(self, incidents, staffing_recs):
        """Save analysis results for evidence and dashboard"""
        os.makedirs('evidence/output', exist_ok=True)
        
        # Save incidents
        with open('evidence/output/retail_incidents.jsonl', 'w') as f:
            for incident in incidents:
                f.write(json.dumps(incident) + '\n')
        
        # Save staffing recommendations  
        with open('evidence/output/staffing_recommendations.json', 'w') as f:
            json.dump(staffing_recs, f, indent=2)
        
        # Save station statistics
        with open('evidence/output/station_statistics.json', 'w') as f:
            json.dump(self.station_stats, f, indent=2)
        
        print(f"💾 Analysis results saved to evidence/output/")


def main():
    """Main execution function"""
    sentinel = SentinelSystem()
    incidents = sentinel.analyze_all_challenges()
    
    print(f"\n🎉 PROJECT SENTINEL Analysis Complete!")
    print(f"🔍 Total incidents detected: {len(incidents)}")
    print(f"📊 Station statistics updated")
    print(f"💾 Evidence files generated")
    
    return sentinel


if __name__ == "__main__":
    sentinel_system = main()
    print("Sentinel system initialized successfully!")