#!/usr/bin/env python3
"""
Startup script for HMM Microservice.

Provides a simple way to start the service with proper configuration.
"""

import os
import sys
import argparse
import uvicorn

from core.config import get_settings


def main():
    """Main entry point for the service."""
    parser = argparse.ArgumentParser(description="HMM Microservice")
    parser.add_argument(
        "--host", 
        default=None, 
        help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None, 
        help="Port to bind to (overrides config)"
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None, 
        help="Number of worker processes"
    )
    parser.add_argument(
        "--log-level", 
        default=None, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    settings = get_settings()
    
    # Override with command line arguments
    host = args.host or settings.host
    port = args.port or settings.port
    workers = args.workers or settings.workers
    log_level = args.log_level or settings.log_level
    
    # Development mode detection
    reload = args.reload or settings.debug
    
    print(f"Starting HMM Microservice on {host}:{port}")
    print(f"Log level: {log_level}")
    print(f"Workers: {workers}")
    print(f"Reload: {reload}")
    
    # Start the server
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()