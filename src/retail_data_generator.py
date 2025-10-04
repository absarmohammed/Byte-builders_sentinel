"""
Retail Data Generator for Byte-builders_sentinel
Generates realistic retail data matching the competition requirements.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid


class RetailDataGenerator:
    """Generates realistic retail data streams for testing"""
    
    def __init__(self):
        self.stations = ["SCC1", "SCC2", "SCC3", "SCC4", "RC1"]  # 4 self-checkout + 1 regular
        self.skus = ["PRD_001", "PRD_002", "PRD_003", "PRD_004", "PRD_005"]
        self.products = {
            "PRD_001": {"name": "Premium Organic Apples", "price": 19.99, "weight": 300, "barcode": "123456789001"},
            "PRD_002": {"name": "Fresh Milk 2L", "price": 12.49, "weight": 250, "barcode": "123456789002"},
            "PRD_003": {"name": "Whole Wheat Bread", "price": 8.99, "weight": 150, "barcode": "123456789003"},
            "PRD_004": {"name": "Greek Yogurt", "price": 15.99, "weight": 200, "barcode": "123456789004"},
            "PRD_005": {"name": "Banana Bundle", "price": 6.49, "weight": 180, "barcode": "123456789005"}
        }
        self.customers = [f"CUST_{1000 + i}" for i in range(50)]
        
    def generate_rfid_data(self, count: int = 20) -> List[Dict[str, Any]]:
        """Generate RFID sensor data"""
        events = []
        base_time = datetime.now()
        
        for i in range(count):
            timestamp = (base_time + timedelta(seconds=i * 5)).isoformat()
            station_id = random.choice(self.stations)
            sku = random.choice(self.skus)
            
            # Introduce some read errors for testing
            status = "Read Error" if random.random() < 0.05 else "Active"
            
            event = {
                "timestamp": timestamp,
                "station_id": station_id,
                "status": status,
                "data": {
                    "epc": f"E{random.randint(200000000000, 299999999999):012d}",
                    "sku": sku,
                    "location": random.choice(["IN_SCAN_AREA", "OUT_SCAN_AREA"])
                }
            }
            events.append(event)
            
        return events
    
    def generate_queue_monitoring_data(self, count: int = 15) -> List[Dict[str, Any]]:
        """Generate queue monitoring system data"""
        events = []
        base_time = datetime.now()
        
        for i in range(count):
            timestamp = (base_time + timedelta(seconds=i * 5)).isoformat()
            station_id = random.choice(self.stations)
            
            # Introduce some read errors for testing
            status = "Read Error" if random.random() < 0.03 else "Active"
            
            # Generate realistic queue data with some problematic situations
            customer_count = random.randint(0, 12)  # Sometimes over the 6-customer target
            base_dwell_time = 45 if customer_count <= 6 else 45 + (customer_count - 6) * 15
            dwell_time = max(20, base_dwell_time + random.randint(-10, 20))
            
            event = {
                "timestamp": timestamp,
                "station_id": station_id,
                "status": status,
                "data": {
                    "customer_count": customer_count,
                    "average_dwell_time": dwell_time
                }
            }
            events.append(event)
            
        return events
    
    def generate_pos_transactions(self, count: int = 25) -> List[Dict[str, Any]]:
        """Generate Point-of-Sale transaction data"""
        events = []
        base_time = datetime.now()
        
        for i in range(count):
            timestamp = (base_time + timedelta(seconds=i * 3)).isoformat()
            station_id = random.choice(self.stations)
            customer_id = random.choice(self.customers)
            sku = random.choice(self.skus)
            product = self.products[sku]
            
            # Introduce system crashes and read errors
            crash_prob = 0.02 if station_id.startswith("SCC") else 0.01  # Self-checkout more prone to crashes
            if random.random() < crash_prob:
                status = "System Crash"
            elif random.random() < 0.03:
                status = "Read Error"
            else:
                status = "Active"
            
            # Introduce price and weight anomalies for barcode switching/weight discrepancy detection
            price = product["price"]
            weight = product["weight"]
            
            # 5% chance of barcode switching (wrong price)
            if random.random() < 0.05:
                wrong_sku = random.choice([s for s in self.skus if s != sku])
                price = self.products[wrong_sku]["price"]
            
            # 3% chance of weight discrepancy
            if random.random() < 0.03:
                weight = weight * random.uniform(0.6, 1.4)  # 40% deviation
            
            event = {
                "timestamp": timestamp,
                "station_id": station_id,
                "status": status,
                "data": {
                    "customer_id": customer_id,
                    "sku": sku,
                    "product_name": product["name"],
                    "barcode": product["barcode"],
                    "price": round(price, 2),
                    "weight_g": round(weight)
                }
            }
            events.append(event)
            
        return events
    
    def generate_product_recognition_data(self, count: int = 15) -> List[Dict[str, Any]]:
        """Generate product recognition analytics data"""
        events = []
        base_time = datetime.now()
        
        for i in range(count):
            timestamp = (base_time + timedelta(seconds=i * 4)).isoformat()
            station_id = random.choice(self.stations)
            
            # Introduce system crashes and read errors
            if random.random() < 0.02:
                status = "System Crash"
            elif random.random() < 0.04:
                status = "Read Error"
            else:
                status = "Active"
            
            # Generate recognition accuracy (sometimes low for testing)
            accuracy = random.uniform(0.7, 0.98)
            
            event = {
                "timestamp": timestamp,
                "station_id": station_id,
                "status": status,
                "data": {
                    "predicted_product": random.choice(self.skus),
                    "accuracy": round(accuracy, 3)
                }
            }
            events.append(event)
            
        return events
    
    def generate_inventory_data(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generate current inventory data"""
        events = []
        base_time = datetime.now()
        
        # Initial inventory
        current_inventory = {
            "PRD_001": 50,
            "PRD_002": 35,
            "PRD_003": 80,
            "PRD_004": 25,
            "PRD_005": 60
        }
        
        for i in range(count):
            timestamp = (base_time + timedelta(minutes=i * 10)).isoformat()
            
            # Simulate inventory changes over time
            for sku in current_inventory:
                # Small random changes, sometimes creating discrepancies
                change = random.randint(-3, 1)  # Slight decline over time
                current_inventory[sku] = max(0, current_inventory[sku] + change)
                
                # Introduce significant discrepancies for testing (5% chance)
                if random.random() < 0.05:
                    current_inventory[sku] = max(0, int(current_inventory[sku] * random.uniform(0.7, 1.3)))
            
            event = {
                "timestamp": timestamp,
                "data": current_inventory.copy()
            }
            events.append(event)
            
        return events
    
    def generate_comprehensive_dataset(self) -> List[Dict[str, Any]]:
        """Generate a comprehensive dataset with all data types"""
        all_events = []
        
        # Generate each type of data
        all_events.extend(self.generate_rfid_data(20))
        all_events.extend(self.generate_queue_monitoring_data(15))
        all_events.extend(self.generate_pos_transactions(25))
        all_events.extend(self.generate_product_recognition_data(15))
        all_events.extend(self.generate_inventory_data(5))
        
        # Sort by timestamp for realistic chronological order
        all_events.sort(key=lambda x: x.get("timestamp", ""))
        
        return all_events
    
    def save_to_jsonl(self, events: List[Dict[str, Any]], filepath: str) -> None:
        """Save events to JSONL file"""
        with open(filepath, 'w') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')


def main():
    """Generate sample retail data"""
    generator = RetailDataGenerator()
    
    # Generate comprehensive dataset
    events = generator.generate_comprehensive_dataset()
    
    # Save to data directory
    from pathlib import Path
    data_dir = Path("data/samples")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    generator.save_to_jsonl(events, str(data_dir / "retail_sample_data.jsonl"))
    
    print(f"Generated {len(events)} retail events")
    print("Data types included:")
    print("- RFID sensor data")
    print("- Queue monitoring data") 
    print("- POS transaction data")
    print("- Product recognition data")
    print("- Inventory tracking data")
    print(f"Saved to: {data_dir / 'retail_sample_data.jsonl'}")


if __name__ == "__main__":
    main()