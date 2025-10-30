#!/usr/bin/env python3
"""
REST API Server for BTC Scalping Model
Provides HTTP endpoint for predictions
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
from predict_live import predict
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for browser access

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'model': 'BTC Scalping Aggressive (15min ±0.08%)',
        'endpoints': {
            '/predict': 'GET - Get current prediction',
            '/health': 'GET - Check API health',
            '/stats': 'GET - Get model statistics'
        }
    })

@app.route('/predict')
def get_prediction():
    """Get current BTC direction prediction."""
    try:
        result = predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': str(pd.Timestamp.now())
    })

@app.route('/stats')
def stats():
    """Get model statistics from config."""
    try:
        from pathlib import Path
        config_path = Path(__file__).parent / 'models/scalping_config.json'
        with open(config_path) as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*80)
    print("🚀 BTC SCALPING API SERVER")
    print("="*80)
    print()
    print("📡 API Endpoints:")
    print("   http://localhost:5000/           - API info")
    print("   http://localhost:5000/predict    - Get prediction")
    print("   http://localhost:5000/health     - Health check")
    print("   http://localhost:5000/stats      - Model stats")
    print()
    print("💡 Usage:")
    print("   curl http://localhost:5000/predict | jq")
    print()
    print("="*80)
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False)



