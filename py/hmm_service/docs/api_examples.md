# HMM Microservice API Examples

This document provides practical examples for using the HMM Microservice API endpoints.

## Table of Contents

- [Authentication](#authentication)
- [Inference Endpoints](#inference-endpoints)
- [Health Check Endpoints](#health-check-endpoints)
- [Model Management Endpoints](#model-management-endpoints)
- [Error Handling](#error-handling)
- [Client Integration Examples](#client-integration-examples)

## Authentication

Currently, no authentication is required. For production deployments, consider implementing:

```bash
# Future API key authentication
curl -H "X-API-Key: your-api-key" http://localhost:8000/inference/predict
```

## Inference Endpoints

### State Probabilities Calculation

Calculate HMM state probabilities for an observation vector:

```bash
# Basic request
curl -X POST http://localhost:8000/inference/state-probabilities \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [0.75, -0.32, 0.18],
    "timestamp": 1640995200
  }'
```

**Response:**
```json
{
  "state_probabilities": [0.15, 0.65, 0.20],
  "most_likely_state": 1,
  "confidence": 0.65,
  "timestamp": 1640995200,
  "processing_time_ms": 12.5,
  "request_id": null
}
```

### Fusion Weights Calculation

Calculate fusion weights for signal combination:

```bash
# Request with tracking ID
curl -X POST http://localhost:8000/inference/fusion-weights \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [0.45, 0.12, -0.67],
    "timestamp": 1640995260,
    "request_id": "trading-signal-001"
  }'
```

**Response:**
```json
{
  "weights": {
    "w_ldc": 0.45,
    "w_mr": 0.30,
    "w_tsmom": 0.25
  },
  "state_probabilities": [0.25, 0.45, 0.30],
  "most_likely_state": 1,
  "timestamp": 1640995260,
  "processing_time_ms": 15.2,
  "request_id": "trading-signal-001"
}
```

### Complete Prediction

Get both state probabilities and fusion weights in one request:

```bash
# Complete prediction request
curl -X POST http://localhost:8000/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [0.82, -0.15, 0.33],
    "timestamp": 1640995320,
    "request_id": "complete-prediction-001"
  }'
```

**Response:**
```json
{
  "state_probabilities": [0.10, 0.70, 0.20],
  "most_likely_state": 1,
  "confidence": 0.70,
  "fusion_weights": {
    "w_ldc": 0.50,
    "w_mr": 0.25,
    "w_tsmom": 0.25
  },
  "timestamp": 1640995320,
  "processing_time_ms": 18.7,
  "model_version": "v1.2.0",
  "request_id": "complete-prediction-001"
}
```

## Health Check Endpoints

### Basic Health Check

Quick health check for load balancers:

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1640995200,
  "uptime": 3600.5,
  "version": "1.0.0"
}
```

### Readiness Check

Comprehensive readiness check for orchestration:

```bash
curl http://localhost:8000/health/ready
```

**Response (Ready):**
```json
{
  "ready": true,
  "model_loaded": true,
  "cache_initialized": true,
  "last_inference": 1640995180,
  "checks": {
    "inference_initialized": true,
    "model_loaded": true,
    "model_valid": true,
    "cache_initialized": true,
    "minio_connected": true,
    "performance_manager_healthy": true
  },
  "timestamp": 1640995200
}
```

**Response (Not Ready - HTTP 503):**
```json
{
  "ready": false,
  "model_loaded": false,
  "cache_initialized": true,
  "last_inference": null,
  "checks": {
    "inference_initialized": false,
    "model_loaded": false,
    "model_valid": false,
    "cache_initialized": true,
    "minio_connected": true,
    "performance_manager_healthy": true
  },
  "timestamp": 1640995200
}
```

### Detailed Health Information

Comprehensive system information:

```bash
curl http://localhost:8000/health/detailed
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1640995200,
  "uptime": 3600.5,
  "version": "1.0.0",
  "memory_usage_mb": 256.7,
  "cpu_usage_percent": 15.2,
  "model_info": {
    "n_states": 3,
    "n_features": 3,
    "has_fusion_weights": true,
    "library": "hmmlearn",
    "training_window": {
      "start_date": "2024-01-01",
      "end_date": "2024-01-31"
    },
    "training_samples": 50000,
    "validation_score": 0.85,
    "artifact_size_mb": 2.5
  },
  "cache_stats": {
    "size": 150,
    "hits": 1250,
    "misses": 350,
    "hit_rate": 0.78
  },
  "performance_stats": {
    "initialized": true,
    "connection_pool": {
      "active_connections": 2,
      "max_connections": 10
    },
    "request_queue": {
      "current_size": 0,
      "max_size": 100
    }
  },
  "config": {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": false,
    "log_level": "INFO",
    "cache_size": 1000,
    "cache_ttl": 300,
    "max_concurrent_requests": 100,
    "default_experiment_id": "production_hmm"
  }
}
```

## Model Management Endpoints

### Reload Model

Hot-reload a model without service downtime:

```bash
# Reload latest version of default experiment
curl -X POST http://localhost:8000/models/reload \
  -H "Content-Type: application/json" \
  -d '{
    "validate_model": true
  }'
```

```bash
# Reload specific experiment and version
curl -X POST http://localhost:8000/models/reload \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_id": "hmm_v2",
    "version": "1.3.0",
    "validate_model": true
  }'
```

**Response:**
```json
{
  "success": true,
  "model_info": {
    "experiment_id": "hmm_v2",
    "version": "1.3.0",
    "load_time": 1640995400,
    "n_states": 3,
    "n_features": 3,
    "has_fusion_weights": true,
    "validation_passed": true
  },
  "reload_time": 2.5,
  "previous_model": {
    "experiment_id": "production_hmm",
    "version": "1.2.0",
    "load_time": 1640995200
  },
  "timestamp": 1640995400
}
```

### Get Current Model Information

Get details about the currently loaded model:

```bash
curl http://localhost:8000/models/current
```

**Response:**
```json
{
  "experiment_id": "production_hmm",
  "version": "1.2.0",
  "load_time": 1640995200,
  "model_info": {
    "n_states": 3,
    "n_features": 3,
    "has_fusion_weights": true,
    "library": "hmmlearn",
    "training_window": {
      "start_date": "2024-01-01",
      "end_date": "2024-01-31"
    },
    "training_samples": 50000,
    "validation_score": 0.85,
    "artifact_size_mb": 2.5
  },
  "performance_stats": {
    "total_inferences": 12500,
    "avg_inference_time_ms": 15.2,
    "error_rate": 0.001,
    "cache_hit_rate": 0.75
  },
  "timestamp": 1640995200
}
```

### List Available Models

Get all available models in storage:

```bash
curl http://localhost:8000/models/available
```

**Response:**
```json
{
  "models": [
    {
      "experiment_id": "production_hmm",
      "version": "1.2.0",
      "created_at": 1640995200,
      "size_mb": 2.5,
      "n_states": 3,
      "n_features": 3,
      "validation_score": 0.85,
      "has_fusion_weights": true,
      "library": "hmmlearn"
    },
    {
      "experiment_id": "production_hmm",
      "version": "1.1.0",
      "created_at": 1640991600,
      "size_mb": 2.3,
      "n_states": 3,
      "n_features": 3,
      "validation_score": 0.82,
      "has_fusion_weights": true,
      "library": "hmmlearn"
    },
    {
      "experiment_id": "hmm_v2",
      "version": "1.3.0",
      "created_at": 1640998800,
      "size_mb": 2.8,
      "n_states": 4,
      "n_features": 3,
      "validation_score": 0.87,
      "has_fusion_weights": true,
      "library": "pomegranate"
    }
  ],
  "total_count": 3,
  "timestamp": 1640995200
}
```

## Error Handling

### Validation Errors (HTTP 400)

```bash
# Invalid observation vector
curl -X POST http://localhost:8000/inference/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [0.75, -0.32]
  }'
```

**Error Response:**
```json
{
  "error": "VALIDATION_ERROR",
  "error_code": "INVALID_OBSERVATIONS",
  "message": "Observations must contain exactly 3 numeric values",
  "timestamp": 1640995200,
  "request_id": null,
  "details": {
    "field": "observations",
    "provided_length": 2,
    "required_length": 3
  }
}
```

### Model Errors (HTTP 404/503)

```bash
# No model loaded
curl http://localhost:8000/models/current
```

**Error Response:**
```json
{
  "error": "MODEL_ERROR",
  "error_code": "NO_MODEL_LOADED",
  "message": "No model is currently loaded",
  "timestamp": 1640995200
}
```

### Service Unavailable (HTTP 503)

```json
{
  "error": "SERVICE_UNAVAILABLE",
  "error_code": "SERVICE_OVERLOADED",
  "message": "Service is temporarily overloaded, please retry",
  "timestamp": 1640995200,
  "details": {
    "retry_after_seconds": 5,
    "current_queue_size": 100,
    "max_queue_size": 100
  }
}
```

## Client Integration Examples

### Python Client

```python
import requests
import json
from typing import Dict, List, Optional

class HMMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
    
    def predict(self, observations: List[float], 
                request_id: Optional[str] = None) -> Dict:
        """Get complete HMM prediction."""
        payload = {
            "observations": observations,
            "timestamp": int(time.time())
        }
        if request_id:
            payload["request_id"] = request_id
        
        response = self.session.post(
            f"{self.base_url}/inference/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_state_probabilities(self, observations: List[float]) -> Dict:
        """Get state probabilities only."""
        payload = {
            "observations": observations,
            "timestamp": int(time.time())
        }
        
        response = self.session.post(
            f"{self.base_url}/inference/state-probabilities",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_fusion_weights(self, observations: List[float]) -> Dict:
        """Get fusion weights only."""
        payload = {
            "observations": observations,
            "timestamp": int(time.time())
        }
        
        response = self.session.post(
            f"{self.base_url}/inference/fusion-weights",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict:
        """Basic health check."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def is_ready(self) -> bool:
        """Check if service is ready."""
        try:
            response = self.session.get(f"{self.base_url}/health/ready")
            return response.status_code == 200 and response.json().get("ready", False)
        except:
            return False

# Usage example
client = HMMClient()

# Check if service is ready
if client.is_ready():
    # Make prediction
    result = client.predict([0.75, -0.32, 0.18], request_id="python-client-001")
    print(f"Most likely state: {result['most_likely_state']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Fusion weights: {result['fusion_weights']}")
else:
    print("Service is not ready")
```

### Rust Client

```rust
use reqwest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize)]
struct InferenceRequest {
    observations: Vec<f64>,
    timestamp: Option<i64>,
    request_id: Option<String>,
}

#[derive(Deserialize)]
struct PredictionResponse {
    state_probabilities: Vec<f64>,
    most_likely_state: usize,
    confidence: f64,
    fusion_weights: HashMap<String, f64>,
    timestamp: i64,
    processing_time_ms: f64,
    model_version: String,
    request_id: Option<String>,
}

pub struct HMMClient {
    client: reqwest::Client,
    base_url: String,
}

impl HMMClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.to_string(),
        }
    }
    
    pub async fn predict(&self, observations: Vec<f64>, request_id: Option<String>) 
        -> Result<PredictionResponse, reqwest::Error> {
        let request = InferenceRequest {
            observations,
            timestamp: Some(chrono::Utc::now().timestamp()),
            request_id,
        };
        
        let response = self.client
            .post(&format!("{}/inference/predict", self.base_url))
            .json(&request)
            .send()
            .await?;
        
        response.json::<PredictionResponse>().await
    }
    
    pub async fn is_ready(&self) -> bool {
        match self.client
            .get(&format!("{}/health/ready", self.base_url))
            .send()
            .await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }
}

// Usage example
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = HMMClient::new("http://localhost:8000");
    
    // Check if service is ready
    if client.is_ready().await {
        // Make prediction
        let result = client.predict(
            vec![0.75, -0.32, 0.18],
            Some("rust-client-001".to_string())
        ).await?;
        
        println!("Most likely state: {}", result.most_likely_state);
        println!("Confidence: {}", result.confidence);
        println!("Fusion weights: {:?}", result.fusion_weights);
    } else {
        println!("Service is not ready");
    }
    
    Ok(())
}
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

class HMMClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }
    
    async predict(observations, requestId = null) {
        const payload = {
            observations,
            timestamp: Math.floor(Date.now() / 1000)
        };
        
        if (requestId) {
            payload.request_id = requestId;
        }
        
        const response = await this.client.post('/inference/predict', payload);
        return response.data;
    }
    
    async getStateProbabilities(observations) {
        const payload = {
            observations,
            timestamp: Math.floor(Date.now() / 1000)
        };
        
        const response = await this.client.post('/inference/state-probabilities', payload);
        return response.data;
    }
    
    async getFusionWeights(observations) {
        const payload = {
            observations,
            timestamp: Math.floor(Date.now() / 1000)
        };
        
        const response = await this.client.post('/inference/fusion-weights', payload);
        return response.data;
    }
    
    async healthCheck() {
        const response = await this.client.get('/health');
        return response.data;
    }
    
    async isReady() {
        try {
            const response = await this.client.get('/health/ready');
            return response.status === 200 && response.data.ready;
        } catch (error) {
            return false;
        }
    }
}

// Usage example
async function main() {
    const client = new HMMClient();
    
    try {
        // Check if service is ready
        if (await client.isReady()) {
            // Make prediction
            const result = await client.predict([0.75, -0.32, 0.18], 'js-client-001');
            
            console.log(`Most likely state: ${result.most_likely_state}`);
            console.log(`Confidence: ${result.confidence}`);
            console.log(`Fusion weights:`, result.fusion_weights);
        } else {
            console.log('Service is not ready');
        }
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

main();
```

## Performance Considerations

### Batch Processing

For high-throughput scenarios, consider batching requests:

```python
import asyncio
import aiohttp

async def batch_predictions(observations_list):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for observations in observations_list:
            task = make_prediction(session, observations)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

async def make_prediction(session, observations):
    payload = {
        "observations": observations,
        "timestamp": int(time.time())
    }
    
    async with session.post(
        "http://localhost:8000/inference/predict",
        json=payload
    ) as response:
        return await response.json()
```

### Connection Pooling

Use connection pooling for better performance:

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure session with connection pooling
session = requests.Session()

# Retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=0.1,
    status_forcelist=[429, 500, 502, 503, 504],
)

# Mount adapter with connection pooling
adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=retry_strategy
)

session.mount("http://", adapter)
session.mount("https://", adapter)
```

### Error Handling and Retries

Implement robust error handling:

```python
import time
import random
from requests.exceptions import RequestException

def make_request_with_retry(client, observations, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.predict(observations)
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
            
            print(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s")
```