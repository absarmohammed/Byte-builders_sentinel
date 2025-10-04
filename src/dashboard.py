"""
Dashboard module for Byte-builders_sentinel
Provides web-based visualization and monitoring interface.
"""

import json
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, jsonify
from pathlib import Path
from typing import Dict, List, Any

app = Flask(__name__)


# @algorithm Data Visualization | Generates interactive charts and graphs for dashboard
def create_event_timeline(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create timeline visualization of events.
    
    Args:
        events: List of events to visualize
        
    Returns:
        Plotly figure dictionary
    """
    if not events:
        return {}
    
    timestamps = [event.get('timestamp', '') for event in events]
    event_types = [event.get('event_type', 'unknown') for event in events]
    
    fig = px.timeline(
        x=timestamps,
        y=event_types,
        title="Event Timeline",
        labels={'x': 'Timestamp', 'y': 'Event Type'}
    )
    
    return fig.to_dict()


# @algorithm Metrics Calculation | Calculates key performance indicators and statistics
def calculate_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate system metrics from events.
    
    Args:
        events: List of events to analyze
        
    Returns:
        Dictionary of calculated metrics
    """
    total_events = len(events)
    event_types = {}
    anomaly_count = 0
    
    for event in events:
        event_type = event.get('event_type', 'unknown')
        event_types[event_type] = event_types.get(event_type, 0) + 1
        
        if 'anomaly' in event_type.lower():
            anomaly_count += 1
    
    metrics = {
        'total_events': total_events,
        'event_types': event_types,
        'anomaly_count': anomaly_count,
        'anomaly_rate': (anomaly_count / total_events * 100) if total_events > 0 else 0
    }
    
    return metrics


@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/events')
def get_events():
    """API endpoint to get events data"""
    # Load events from output files
    events = []
    
    test_file = Path('evidence/output/test/events.jsonl')
    final_file = Path('evidence/output/final/events.jsonl')
    
    for file_path in [test_file, final_file]:
        if file_path.exists():
            with open(file_path, 'r') as f:
                events.extend([json.loads(line) for line in f if line.strip()])
    
    return jsonify(events)


@app.route('/api/metrics')
def get_metrics():
    """API endpoint to get system metrics"""
    # Load and calculate metrics
    events = []
    
    test_file = Path('evidence/output/test/events.jsonl')
    final_file = Path('evidence/output/final/events.jsonl')
    
    for file_path in [test_file, final_file]:
        if file_path.exists():
            with open(file_path, 'r') as f:
                events.extend([json.loads(line) for line in f if line.strip()])
    
    metrics = calculate_metrics(events)
    return jsonify(metrics)


@app.route('/api/timeline')
def get_timeline():
    """API endpoint to get timeline visualization"""
    # Load events and create timeline
    events = []
    
    test_file = Path('evidence/output/test/events.jsonl')
    final_file = Path('evidence/output/final/events.jsonl')
    
    for file_path in [test_file, final_file]:
        if file_path.exists():
            with open(file_path, 'r') as f:
                events.extend([json.loads(line) for line in f if line.strip()])
    
    timeline = create_event_timeline(events)
    return jsonify(timeline)


if __name__ == '__main__':
    # Create templates directory and basic template
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Byte-builders Sentinel Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .metrics { display: flex; gap: 20px; margin-bottom: 30px; }
        .metric-card { padding: 20px; background: #f5f5f5; border-radius: 8px; flex: 1; }
        .chart-container { margin-bottom: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Byte-builders Sentinel Dashboard</h1>
        
        <div class="metrics" id="metrics">
            <!-- Metrics will be loaded here -->
        </div>
        
        <div class="chart-container">
            <div id="timeline-chart"></div>
        </div>
    </div>
    
    <script>
        // Load metrics
        fetch('/api/metrics')
            .then(response => response.json())
            .then(data => {
                const metricsContainer = document.getElementById('metrics');
                metricsContainer.innerHTML = `
                    <div class="metric-card">
                        <h3>Total Events</h3>
                        <p style="font-size: 2em; margin: 0;">${data.total_events}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Anomalies</h3>
                        <p style="font-size: 2em; margin: 0;">${data.anomaly_count}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Anomaly Rate</h3>
                        <p style="font-size: 2em; margin: 0;">${data.anomaly_rate.toFixed(1)}%</p>
                    </div>
                `;
            });
        
        // Load timeline chart
        fetch('/api/timeline')
            .then(response => response.json())
            .then(data => {
                if (Object.keys(data).length > 0) {
                    Plotly.newPlot('timeline-chart', data.data, data.layout);
                }
            });
    </script>
</body>
</html>
    """
    
    with open(templates_dir / 'dashboard.html', 'w') as f:
        f.write(dashboard_html)
    
    app.run(debug=True, port=5000)