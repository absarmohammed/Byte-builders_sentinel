"""
PROJECT SENTINEL - Machine Learning Models
Advanced ML-based fraud detection and retail analytics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentinelMLEngine:
    """
    PROJECT SENTINEL - Machine Learning Engine
    Comprehensive ML suite for retail fraud detection and analytics
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_performance = {}
        
        # Initialize model configurations
        self.model_configs = {
            'fraud_detection': {
                'algorithm': 'RandomForest',
                'features': ['price_ratio', 'weight_ratio', 'time_anomaly', 'customer_risk', 'product_risk'],
                'target': 'is_fraud'
            },
            'customer_behavior': {
                'algorithm': 'IsolationForest',
                'features': ['spending_avg', 'frequency', 'high_value_ratio', 'age', 'session_duration'],
                'anomaly_threshold': 0.1
            },
            'queue_prediction': {
                'algorithm': 'LinearRegression',
                'features': ['hour', 'day_of_week', 'transaction_volume', 'station_count'],
                'target': 'queue_length'
            },
            'inventory_shrinkage': {
                'algorithm': 'DBSCAN',
                'features': ['sales_variance', 'rfid_variance', 'time_gap', 'product_category'],
                'clustering_params': {'eps': 0.5, 'min_samples': 5}
            }
        }
        
        print("🤖 PROJECT SENTINEL ML Engine Initialized")
        print("📊 Models: Fraud Detection | Customer Behavior | Queue Prediction | Inventory Analytics")
    
    def prepare_fraud_detection_data(self, pos_data, recognition_data, customer_db, product_catalog):
        """
        Prepare training data for fraud detection model
        """
        features = []
        labels = []
        
        for pos_tx in pos_data:
            pos_data_obj = pos_tx.get('data', {})
            customer_id = pos_data_obj.get('customer_id')
            sku = pos_data_obj.get('sku')
            pos_price = pos_data_obj.get('price', 0)
            timestamp = pos_tx.get('timestamp')
            
            if customer_id and sku:
                # Find corresponding recognition event
                recognition_match = None
                for rec_event in recognition_data:
                    if (rec_event.get('station_id') == pos_tx.get('station_id') and 
                        abs(self._time_diff(rec_event['timestamp'], timestamp)) <= 10):
                        recognition_match = rec_event
                        break
                
                if recognition_match:
                    rec_data = recognition_match.get('data', {})
                    recognized_sku = rec_data.get('predicted_product')
                    accuracy = rec_data.get('accuracy', 0)
                    
                    # Get customer and product info
                    customer_info = customer_db.get(customer_id, {})
                    product_info = product_catalog.get(sku, {})
                    recognized_product = product_catalog.get(recognized_sku, {}) if recognized_sku else {}
                    
                    # Calculate features
                    price_ratio = (recognized_product.get('price', pos_price) / pos_price) if pos_price > 0 else 1.0
                    weight_ratio = (product_info.get('weight', 100) / 100)  # Normalized weight
                    time_anomaly = self._calculate_time_anomaly(timestamp)
                    customer_risk = customer_info.get('risk_score', 0.5)
                    product_risk = product_info.get('theft_risk', 0.5)
                    
                    # Additional features
                    recognition_confidence = accuracy
                    price_difference = abs(recognized_product.get('price', pos_price) - pos_price)
                    is_high_value = 1 if pos_price > 500 else 0
                    
                    feature_vector = [
                        price_ratio,
                        weight_ratio, 
                        time_anomaly,
                        customer_risk,
                        product_risk,
                        recognition_confidence,
                        price_difference,
                        is_high_value
                    ]
                    
                    # Label: 1 if fraud (significant price difference), 0 otherwise
                    is_fraud = 1 if (recognized_sku and recognized_sku != sku and 
                                   price_ratio > 1.5 and accuracy > 0.8) else 0
                    
                    features.append(feature_vector)
                    labels.append(is_fraud)
        
        return np.array(features), np.array(labels)
    
    def prepare_customer_behavior_data(self, pos_data, customer_db):
        """
        Prepare data for customer behavior anomaly detection
        """
        customer_features = {}
        
        # Aggregate customer data
        for pos_tx in pos_data:
            customer_id = pos_tx.get('data', {}).get('customer_id')
            price = pos_tx.get('data', {}).get('price', 0)
            
            if customer_id:
                if customer_id not in customer_features:
                    customer_features[customer_id] = {
                        'transactions': [],
                        'total_spent': 0,
                        'high_value_purchases': 0
                    }
                
                customer_features[customer_id]['transactions'].append(price)
                customer_features[customer_id]['total_spent'] += price
                if price > 500:
                    customer_features[customer_id]['high_value_purchases'] += 1
        
        # Convert to feature matrix
        features = []
        customer_ids = []
        
        for customer_id, data in customer_features.items():
            if len(data['transactions']) >= 3:  # Minimum transactions for analysis
                customer_info = customer_db.get(customer_id, {})
                
                spending_avg = np.mean(data['transactions'])
                spending_std = np.std(data['transactions'])
                frequency = len(data['transactions'])
                high_value_ratio = data['high_value_purchases'] / frequency
                age = customer_info.get('age', 35)
                
                feature_vector = [
                    spending_avg,
                    spending_std,
                    frequency,
                    high_value_ratio,
                    age,
                    data['total_spent']
                ]
                
                features.append(feature_vector)
                customer_ids.append(customer_id)
        
        return np.array(features), customer_ids
    
    def prepare_queue_prediction_data(self, queue_data, pos_data):
        """
        Prepare data for queue length prediction
        """
        # Aggregate data by hour and station
        hourly_data = {}
        
        for queue_event in queue_data:
            timestamp = datetime.fromisoformat(queue_event['timestamp'].replace('Z', ''))
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            station_id = queue_event.get('station_id')
            queue_length = queue_event.get('data', {}).get('customer_count', 0)
            
            key = f"{hour}_{day_of_week}_{station_id}"
            if key not in hourly_data:
                hourly_data[key] = {
                    'queue_lengths': [],
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'station_id': station_id
                }
            hourly_data[key]['queue_lengths'].append(queue_length)
        
        # Count transactions per hour
        transaction_counts = {}
        for pos_tx in pos_data:
            timestamp = datetime.fromisoformat(pos_tx['timestamp'].replace('Z', ''))
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            station_id = pos_tx.get('station_id')
            
            key = f"{hour}_{day_of_week}_{station_id}"
            transaction_counts[key] = transaction_counts.get(key, 0) + 1
        
        # Create feature matrix
        features = []
        targets = []
        
        for key, data in hourly_data.items():
            if len(data['queue_lengths']) > 0:
                avg_queue_length = np.mean(data['queue_lengths'])
                transaction_volume = transaction_counts.get(key, 0)
                
                feature_vector = [
                    data['hour'],
                    data['day_of_week'],
                    transaction_volume,
                    1  # station_count (simplified)
                ]
                
                features.append(feature_vector)
                targets.append(avg_queue_length)
        
        return np.array(features), np.array(targets)
    
    def train_fraud_detection_model(self, pos_data, recognition_data, customer_db, product_catalog):
        """
        Train Random Forest model for fraud detection
        """
        print("🔍 Training Fraud Detection Model...")
        
        # Prepare data
        X, y = self.prepare_fraud_detection_data(pos_data, recognition_data, customer_db, product_catalog)
        
        if len(X) < 10:
            print("⚠️ Insufficient data for fraud detection training")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = model.score(X_test_scaled, y_test)
        
        # Store model
        self.models['fraud_detection'] = model
        self.scalers['fraud_detection'] = scaler
        self.model_performance['fraud_detection'] = {
            'accuracy': accuracy,
            'fraud_cases_detected': sum(y_pred),
            'total_test_cases': len(y_test),
            'feature_importance': model.feature_importances_.tolist()
        }
        
        print(f"✅ Fraud Detection Model Trained - Accuracy: {accuracy:.3f}")
        print(f"📊 Detected {sum(y_pred)} potential fraud cases out of {len(y_test)} test cases")
        
        return True
    
    def train_customer_behavior_model(self, pos_data, customer_db):
        """
        Train Isolation Forest for customer behavior anomaly detection
        """
        print("👤 Training Customer Behavior Model...")
        
        # Prepare data
        X, customer_ids = self.prepare_customer_behavior_data(pos_data, customer_db)
        
        if len(X) < 5:
            print("⚠️ Insufficient customer data for behavior analysis")
            return False
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = model.fit_predict(X_scaled)
        
        # Store model
        self.models['customer_behavior'] = model
        self.scalers['customer_behavior'] = scaler
        self.model_performance['customer_behavior'] = {
            'anomalies_detected': sum(anomaly_scores == -1),
            'total_customers': len(customer_ids),
            'anomaly_rate': sum(anomaly_scores == -1) / len(customer_ids)
        }
        
        print(f"✅ Customer Behavior Model Trained")
        print(f"📊 Detected {sum(anomaly_scores == -1)} anomalous customers out of {len(customer_ids)}")
        
        return True
    
    def train_queue_prediction_model(self, queue_data, pos_data):
        """
        Train Linear Regression for queue length prediction
        """
        print("🚶 Training Queue Prediction Model...")
        
        # Prepare data
        X, y = self.prepare_queue_prediction_data(queue_data, pos_data)
        
        if len(X) < 10:
            print("⚠️ Insufficient queue data for prediction training")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store model
        self.models['queue_prediction'] = model
        self.scalers['queue_prediction'] = scaler
        self.model_performance['queue_prediction'] = {
            'mse': mse,
            'r2_score': r2,
            'mean_actual_queue': np.mean(y_test),
            'mean_predicted_queue': np.mean(y_pred)
        }
        
        print(f"✅ Queue Prediction Model Trained - R² Score: {r2:.3f}")
        print(f"📊 MSE: {mse:.3f}, Avg Queue Length: {np.mean(y_test):.1f}")
        
        return True
    
    def predict_fraud_risk(self, transaction_data):
        """
        Predict fraud risk for a transaction
        """
        if 'fraud_detection' not in self.models:
            return {'risk_score': 0.5, 'confidence': 0.0, 'status': 'model_not_trained'}
        
        model = self.models['fraud_detection']
        scaler = self.scalers['fraud_detection']
        
        # Prepare feature vector (simplified for example)
        features = np.array([[
            transaction_data.get('price_ratio', 1.0),
            transaction_data.get('weight_ratio', 1.0),
            transaction_data.get('time_anomaly', 0.0),
            transaction_data.get('customer_risk', 0.5),
            transaction_data.get('product_risk', 0.5),
            transaction_data.get('recognition_confidence', 0.8),
            transaction_data.get('price_difference', 0.0),
            transaction_data.get('is_high_value', 0)
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        risk_probability = model.predict_proba(features_scaled)[0]
        
        return {
            'risk_score': risk_probability[1],  # Probability of fraud
            'confidence': max(risk_probability),
            'status': 'prediction_ready'
        }
    
    def detect_customer_anomaly(self, customer_data):
        """
        Detect if customer behavior is anomalous
        """
        if 'customer_behavior' not in self.models:
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'status': 'model_not_trained'}
        
        model = self.models['customer_behavior']
        scaler = self.scalers['customer_behavior']
        
        # Prepare feature vector
        features = np.array([[
            customer_data.get('spending_avg', 300),
            customer_data.get('spending_std', 100),
            customer_data.get('frequency', 5),
            customer_data.get('high_value_ratio', 0.1),
            customer_data.get('age', 35),
            customer_data.get('total_spent', 1500)
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        anomaly_prediction = model.predict(features_scaled)[0]
        anomaly_score = model.decision_function(features_scaled)[0]
        
        return {
            'is_anomaly': anomaly_prediction == -1,
            'anomaly_score': anomaly_score,
            'status': 'prediction_ready'
        }
    
    def predict_queue_length(self, hour, day_of_week, transaction_volume):
        """
        Predict expected queue length
        """
        if 'queue_prediction' not in self.models:
            return {'predicted_queue': 2.0, 'confidence': 0.0, 'status': 'model_not_trained'}
        
        model = self.models['queue_prediction']
        scaler = self.scalers['queue_prediction']
        
        # Prepare feature vector
        features = np.array([[hour, day_of_week, transaction_volume, 1]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        predicted_queue = model.predict(features_scaled)[0]
        
        return {
            'predicted_queue': max(0, predicted_queue),  # Queue can't be negative
            'confidence': 0.8,  # Simplified confidence
            'status': 'prediction_ready'
        }
    
    def save_models(self, model_dir="models"):
        """
        Save trained models to disk
        """
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_file = os.path.join(model_dir, f"{model_name}_model.pkl")
            scaler_file = os.path.join(model_dir, f"{model_name}_scaler.pkl")
            
            joblib.dump(model, model_file)
            if model_name in self.scalers:
                joblib.dump(self.scalers[model_name], scaler_file)
        
        # Save performance metrics
        performance_file = os.path.join(model_dir, "model_performance.json")
        with open(performance_file, 'w') as f:
            json.dump(self.model_performance, f, indent=2)
        
        print(f"💾 Models saved to {model_dir}/")
    
    def load_models(self, model_dir="models"):
        """
        Load trained models from disk
        """
        if not os.path.exists(model_dir):
            print(f"⚠️ Model directory {model_dir} not found")
            return False
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('_model.pkl', '')
            model_path = os.path.join(model_dir, model_file)
            scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
            
            self.models[model_name] = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scalers[model_name] = joblib.load(scaler_path)
        
        # Load performance metrics
        performance_file = os.path.join(model_dir, "model_performance.json")
        if os.path.exists(performance_file):
            with open(performance_file, 'r') as f:
                self.model_performance = json.load(f)
        
        print(f"📁 Loaded {len(self.models)} models from {model_dir}/")
        return True
    
    def _time_diff(self, time1, time2):
        """Calculate time difference in seconds"""
        try:
            t1 = datetime.fromisoformat(time1.replace('Z', ''))
            t2 = datetime.fromisoformat(time2.replace('Z', ''))
            return (t1 - t2).total_seconds()
        except:
            return 0
    
    def _calculate_time_anomaly(self, timestamp):
        """Calculate time-based anomaly score"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', ''))
            hour = dt.hour
            
            # Business hours anomaly (higher risk outside 9-17)
            if hour < 9 or hour > 17:
                return 0.8
            elif hour < 11 or hour > 15:
                return 0.3
            else:
                return 0.1
        except:
            return 0.5
    
    def get_model_summary(self):
        """
        Get comprehensive summary of all trained models
        """
        summary = {
            'models_trained': len(self.models),
            'model_details': {},
            'training_status': 'complete' if len(self.models) > 0 else 'pending'
        }
        
        for model_name, performance in self.model_performance.items():
            summary['model_details'][model_name] = {
                'status': 'trained',
                'performance': performance,
                'model_type': self.model_configs.get(model_name, {}).get('algorithm', 'Unknown')
            }
        
        return summary


# Integration with main SentinelSystem
class MLEnhancedSentinelSystem:
    """
    Enhanced Sentinel System with ML capabilities
    """
    
    def __init__(self, sentinel_system, ml_engine):
        self.sentinel = sentinel_system
        self.ml_engine = ml_engine
        self.ml_predictions = []
    
    def train_all_models(self):
        """
        Train all ML models with current data
        """
        print("\n🤖 TRAINING ML MODELS FOR ENHANCED ANALYTICS")
        print("=" * 60)
        
        # Load data
        data = self.sentinel.load_data()
        
        # Train fraud detection
        self.ml_engine.train_fraud_detection_model(
            data['pos'], data['recognition'], 
            self.sentinel.customer_database, self.sentinel.product_catalog
        )
        
        # Train customer behavior analysis
        self.ml_engine.train_customer_behavior_model(
            data['pos'], self.sentinel.customer_database
        )
        
        # Train queue prediction
        self.ml_engine.train_queue_prediction_model(
            data['queue'], data['pos']
        )
        
        # Save models
        self.ml_engine.save_models()
        
        return self.ml_engine.get_model_summary()
    
    def get_ml_enhanced_analysis(self):
        """
        Get ML-enhanced analysis results
        """
        print("\n🧠 ML-ENHANCED REAL-TIME PREDICTIONS")
        print("=" * 50)
        
        predictions = []
        
        # Sample fraud prediction
        fraud_prediction = self.ml_engine.predict_fraud_risk({
            'price_ratio': 2.5,
            'customer_risk': 0.7,
            'product_risk': 0.8,
            'recognition_confidence': 0.9
        })
        
        if fraud_prediction['status'] == 'prediction_ready':
            predictions.append({
                'type': 'fraud_risk_prediction',
                'risk_score': fraud_prediction['risk_score'],
                'confidence': fraud_prediction['confidence'],
                'recommendation': 'high_priority_review' if fraud_prediction['risk_score'] > 0.7 else 'standard_monitoring'
            })
            print(f"🚨 Fraud Risk Prediction: {fraud_prediction['risk_score']:.3f} (Confidence: {fraud_prediction['confidence']:.3f})")
        
        # Sample customer anomaly detection
        customer_anomaly = self.ml_engine.detect_customer_anomaly({
            'spending_avg': 800,
            'frequency': 15,
            'high_value_ratio': 0.6,
            'age': 25
        })
        
        if customer_anomaly['status'] == 'prediction_ready':
            predictions.append({
                'type': 'customer_behavior_anomaly',
                'is_anomaly': customer_anomaly['is_anomaly'],
                'anomaly_score': customer_anomaly['anomaly_score'],
                'recommendation': 'investigate_customer' if customer_anomaly['is_anomaly'] else 'normal_behavior'
            })
            print(f"👤 Customer Anomaly: {'Yes' if customer_anomaly['is_anomaly'] else 'No'} (Score: {customer_anomaly['anomaly_score']:.3f})")
        
        # Sample queue prediction
        queue_prediction = self.ml_engine.predict_queue_length(14, 2, 25)  # 2PM, Wednesday, 25 transactions
        
        if queue_prediction['status'] == 'prediction_ready':
            predictions.append({
                'type': 'queue_length_prediction',
                'predicted_queue': queue_prediction['predicted_queue'],
                'confidence': queue_prediction['confidence'],
                'recommendation': 'open_additional_station' if queue_prediction['predicted_queue'] > 5 else 'maintain_current_staffing'
            })
            print(f"🚶 Queue Prediction: {queue_prediction['predicted_queue']:.1f} customers (Confidence: {queue_prediction['confidence']:.3f})")
        
        self.ml_predictions = predictions
        return predictions


def main():
    """
    Main function to demonstrate ML capabilities
    """
    # Initialize ML Engine
    ml_engine = SentinelMLEngine()
    
    # This would typically integrate with the main SentinelSystem
    print("\n" + "="*80)
    print("🤖 PROJECT SENTINEL - MACHINE LEARNING DEMONSTRATION")
    print("="*80)
    
    print("\n📋 ML Model Configurations:")
    for model_name, config in ml_engine.model_configs.items():
        print(f"   • {model_name}: {config['algorithm']}")
        print(f"     Features: {', '.join(config['features'])}")
    
    print(f"\n✅ ML Engine initialized with {len(ml_engine.model_configs)} model types")
    print("🔄 Ready for integration with main SentinelSystem")
    
    return ml_engine


if __name__ == "__main__":
    ml_engine = main()