#!/bin/bash

# Integration Test Runner for Training Data CLI
# This script runs all integration tests with proper setup and reporting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ] || [ ! -d "tests" ]; then
    print_error "Please run this script from the training-data-cli directory"
    exit 1
fi

print_status "Starting integration tests for Training Data CLI"

# Build the binary first
print_status "Building training-data binary..."
if cargo build; then
    print_success "Binary built successfully"
else
    print_error "Failed to build binary"
    exit 1
fi

# Check for sample data
SAMPLE_DATA="../sample/ohlcv.parquet"
if [ ! -f "$SAMPLE_DATA" ]; then
    print_warning "Sample data not found at $SAMPLE_DATA"
    print_warning "Some tests may fail or be skipped"
fi

# Create test results directory
TEST_RESULTS_DIR="test_results"
mkdir -p "$TEST_RESULTS_DIR"

print_status "Running integration tests..."

# Function to run a test suite
run_test_suite() {
    local test_name=$1
    local test_file=$2
    local description=$3
    
    print_status "Running $description..."
    
    if cargo test --test "$test_file" -- --nocapture > "$TEST_RESULTS_DIR/${test_name}_results.txt" 2>&1; then
        print_success "$description completed successfully"
        return 0
    else
        print_warning "$description had some failures (check $TEST_RESULTS_DIR/${test_name}_results.txt)"
        return 1
    fi
}

# Track test results
TOTAL_SUITES=0
PASSED_SUITES=0

# Run integration test suites
print_status "=== Running Integration Test Suites ==="

# End-to-end integration tests
TOTAL_SUITES=$((TOTAL_SUITES + 1))
if run_test_suite "integration" "integration_tests" "End-to-end integration tests"; then
    PASSED_SUITES=$((PASSED_SUITES + 1))
fi

# CLI integration tests
TOTAL_SUITES=$((TOTAL_SUITES + 1))
if run_test_suite "cli" "cli_integration_tests" "CLI integration tests"; then
    PASSED_SUITES=$((PASSED_SUITES + 1))
fi

# Performance tests (run separately due to resource requirements)
print_status "=== Running Performance Tests ==="
TOTAL_SUITES=$((TOTAL_SUITES + 1))
if run_test_suite "performance" "performance_tests" "Performance benchmark tests"; then
    PASSED_SUITES=$((PASSED_SUITES + 1))
fi

# Generate summary report
print_status "=== Test Summary ==="
echo "Test Results Summary" > "$TEST_RESULTS_DIR/summary.txt"
echo "===================" >> "$TEST_RESULTS_DIR/summary.txt"
echo "Total test suites: $TOTAL_SUITES" >> "$TEST_RESULTS_DIR/summary.txt"
echo "Passed: $PASSED_SUITES" >> "$TEST_RESULTS_DIR/summary.txt"
echo "Failed: $((TOTAL_SUITES - PASSED_SUITES))" >> "$TEST_RESULTS_DIR/summary.txt"
echo "Success rate: $(( (PASSED_SUITES * 100) / TOTAL_SUITES ))%" >> "$TEST_RESULTS_DIR/summary.txt"

# Display summary
cat "$TEST_RESULTS_DIR/summary.txt"

# Check for specific test results
print_status "=== Detailed Results ==="

# Check for performance metrics
if [ -f "$TEST_RESULTS_DIR/performance_results.txt" ]; then
    print_status "Performance test metrics:"
    grep -E "(rows/sec|MB|Duration|Throughput)" "$TEST_RESULTS_DIR/performance_results.txt" | head -10
fi

# Check for validation results
if [ -f "$TEST_RESULTS_DIR/integration_results.txt" ]; then
    print_status "Integration test highlights:"
    grep -E "(SUCCESS|WARNING|ERROR)" "$TEST_RESULTS_DIR/integration_results.txt" | tail -5
fi

# Final status
if [ $PASSED_SUITES -eq $TOTAL_SUITES ]; then
    print_success "All test suites completed successfully!"
    exit 0
elif [ $PASSED_SUITES -gt 0 ]; then
    print_warning "Some test suites had issues. Check individual result files for details."
    exit 1
else
    print_error "All test suites failed. Check result files for details."
    exit 1
fi