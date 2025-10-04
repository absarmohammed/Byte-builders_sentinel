# Byte-builders_sentinel

## Development

### Setup
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
python -m pytest src/test_sentinel.py -v
```

### Running the Application
```bash
python src/main.py
```

### Running the Dashboard
```bash
python src/dashboard.py
```

## Project Structure
- `src/main.py` - Core application logic with algorithm implementations
- `src/dashboard.py` - Web dashboard for visualization
- `src/utils.py` - Utility functions and helper classes
- `src/test_sentinel.py` - Unit tests
- `config.json` - Application configuration
- `requirements.txt` - Python dependencies

## Algorithm Documentation

The following algorithms are implemented with proper tagging:

1. **Event Processing** (`@algorithm Event Processing | Processes incoming events and generates structured output`)
   - Located in `src/main.py:process_events()`
   - Purpose: Transform raw input events into structured output format

2. **Anomaly Detection** (`@algorithm Anomaly Detection | Detects anomalous patterns in event data`)
   - Located in `src/main.py:detect_anomalies()`
   - Purpose: Identify unusual patterns or threshold violations in event streams

3. **Data Visualization** (`@algorithm Data Visualization | Generates interactive charts and graphs for dashboard`)
   - Located in `src/dashboard.py:create_event_timeline()`
   - Purpose: Create visual representations of event data for dashboard

4. **Metrics Calculation** (`@algorithm Metrics Calculation | Calculates key performance indicators and statistics`)
   - Located in `src/dashboard.py:calculate_metrics()`
   - Purpose: Compute system performance metrics and statistics

5. **Data Validation** (`@algorithm Data Validation | Validates input data structure and content`)
   - Located in `src/utils.py:validate_event_data()`
   - Purpose: Ensure data integrity and format compliance

6. **File Processing** (`@algorithm File Processing | Handles file I/O operations for various data formats`)
   - Located in `src/utils.py:load_jsonl_file()`
   - Purpose: Efficient reading and writing of JSONL data files

7. **Configuration Management** (`@algorithm Configuration Management | Handles application configuration and settings`)
   - Located in `src/utils.py:load_config()`
   - Purpose: Manage application settings and configuration parameters