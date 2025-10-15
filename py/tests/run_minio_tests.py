#!/usr/bin/env python3
"""
Comprehensive test runner for MinIO artifact storage functionality.

This script runs all MinIO-related tests:
1. Unit tests (with mocked MinIO client)
2. Integration tests (with real MinIO instance)

Usage:
    # Run all tests (unit + integration if MinIO available)
    python tests/run_minio_tests.py
    
    # Run only unit tests
    python tests/run_minio_tests.py --unit-only
    
    # Run only integration tests
    python tests/run_minio_tests.py --integration-only
    
    # Run with verbose output
    python tests/run_minio_tests.py -v
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run MinIO artifact storage tests")
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only unit tests (with mocked MinIO)"
    )
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests (requires MinIO)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Build pytest options
    pytest_opts = "-v" if args.verbose else ""
    if args.coverage:
        pytest_opts += " --cov=imp.hmm.artifact_management --cov-report=term-missing"
    
    results = []
    
    # Run unit tests
    if not args.integration_only:
        print("\n" + "="*70)
        print("  RUNNING UNIT TESTS")
        print("="*70)
        print("\nThese tests use mocked MinIO client and don't require a running MinIO instance.")
        
        unit_tests = [
            "tests/test_minio_artifact_store.py",
            "tests/test_experiment_tracker_minio.py"
        ]
        
        for test_file in unit_tests:
            cmd = f"python -m pytest {test_file} {pytest_opts}"
            success = run_command(cmd, f"Unit Tests: {test_file}")
            results.append(("Unit Tests", test_file, success))
    
    # Run integration tests
    if not args.unit_only:
        print("\n" + "="*70)
        print("  RUNNING INTEGRATION TESTS")
        print("="*70)
        print("\nThese tests require a running MinIO instance.")
        print("If MinIO is not available, tests will be skipped.")
        print("\nTo start MinIO:")
        print("  docker-compose up -d")
        print()
        
        cmd = f"python -m pytest tests/test_minio_integration.py -m integration {pytest_opts}"
        success = run_command(cmd, "Integration Tests: test_minio_integration.py")
        results.append(("Integration Tests", "test_minio_integration.py", success))
    
    # Print summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_type, test_file, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_type} - {test_file}")
        if not success:
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed. See output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
