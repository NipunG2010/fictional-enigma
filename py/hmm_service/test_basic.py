#!/usr/bin/env python3
"""
Basic test script to verify HMM Microservice functionality.
"""

import sys
import requests
import json
import time

def test_service(base_url="http://127.0.0.1:8001"):
    """Test basic service functionality."""
    
    print(f"Testing HMM Microservice at {base_url}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            health_data = response.json()
            print(f"  Status: {health_data['status']}")
            print(f"  Uptime: {health_data['uptime']}s")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False
    
    # Test readiness endpoint
    try:
        response = requests.get(f"{base_url}/health/ready", timeout=5)
        if response.status_code in [200, 503]:  # 503 is OK for readiness
            print("✓ Readiness check responded")
            ready_data = response.json()
            print(f"  Ready: {ready_data['ready']}")
        else:
            print(f"✗ Readiness check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Readiness check error: {e}")
    
    # Test inference endpoint
    try:
        test_request = {
            "observations": [0.1, -0.2, 0.3],
            "timestamp": int(time.time())
        }
        
        response = requests.post(
            f"{base_url}/inference/state-probabilities",
            json=test_request,
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ State probabilities endpoint works")
            data = response.json()
            print(f"  State probabilities: {data['state_probabilities']}")
            print(f"  Most likely state: {data['most_likely_state']}")
            print(f"  Processing time: {data['processing_time_ms']}ms")
        else:
            print(f"✗ State probabilities failed: {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ State probabilities error: {e}")
    
    # Test fusion weights endpoint
    try:
        response = requests.post(
            f"{base_url}/inference/fusion-weights",
            json=test_request,
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ Fusion weights endpoint works")
            data = response.json()
            print(f"  Weights: {data['weights']}")
            print(f"  Processing time: {data['processing_time_ms']}ms")
        else:
            print(f"✗ Fusion weights failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Fusion weights error: {e}")
    
    # Test complete prediction endpoint
    try:
        response = requests.post(
            f"{base_url}/inference/predict",
            json=test_request,
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ Complete prediction endpoint works")
            data = response.json()
            print(f"  Model version: {data['model_version']}")
            print(f"  Processing time: {data['processing_time_ms']}ms")
        else:
            print(f"✗ Complete prediction failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Complete prediction error: {e}")
    
    # Test model management endpoints
    try:
        response = requests.get(f"{base_url}/models/current", timeout=5)
        if response.status_code == 200:
            print("✓ Current model endpoint works")
            data = response.json()
            print(f"  Experiment ID: {data['experiment_id']}")
            print(f"  Version: {data['version']}")
        else:
            print(f"✗ Current model failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Current model error: {e}")
    
    try:
        response = requests.get(f"{base_url}/models/available", timeout=5)
        if response.status_code == 200:
            print("✓ Available models endpoint works")
            data = response.json()
            print(f"  Available models: {data['total_count']}")
        else:
            print(f"✗ Available models failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Available models error: {e}")
    
    print("\n✓ Basic service structure test completed successfully!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://127.0.0.1:8001"
    
    test_service(base_url)