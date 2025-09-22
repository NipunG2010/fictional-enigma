#!/bin/bash

# Comprehensive Performance Testing Script for LDC Engine
# This script runs all performance optimization tests and generates a detailed report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
RESULTS_DIR="performance_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}🚀 Starting Comprehensive Performance Testing for LDC Engine${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# System information
echo -e "${YELLOW}📊 System Information:${NC}"
echo "  Date: $(date)"
echo "  Hostname: $(hostname)"
echo "  CPU: $(nproc) cores"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "  Rust version: $(rustc --version)"
echo ""

# Function to run tests and capture output
run_test_suite() {
    local test_name="$1"
    local test_command="$2"
    local log_file="$RESULTS_DIR/${test_name}_${TIMESTAMP}.log"
    
    echo -e "${YELLOW}🧪 Running $test_name...${NC}"
    
    if eval "$test_command" > "$log_file" 2>&1; then
        echo -e "${GREEN}✅ $test_name: PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ $test_name: FAILED${NC}"
        echo "   Log: $log_file"
        return 1
    fi
}

# Initialize counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test 1: Unit Tests
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test_suite "Unit Tests" "cargo test --lib --release"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 2: Performance Requirements Tests
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test_suite "Performance Requirements" "cargo test --test performance_requirements_tests --release"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 3: Performance Validation Tests
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test_suite "Performance Validation" "cargo test --test performance_validation_tests --release"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 4: Pine Script Compatibility Tests
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test_suite "Pine Script Compatibility" "cargo test --test pine_script_compatibility_tests --release"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 5: Comprehensive Integration Tests (selected tests)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test_suite "SIMD Compatibility" "cargo test --test comprehensive_integration_tests test_simd_pine_script_compatibility_comprehensive --release"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 6: Memory Tests
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test_suite "Memory Tests" "cargo test --test comprehensive_integration_tests test_memory_usage_patterns_and_mapping --release"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 7: Stress Tests
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test_suite "Stress Tests" "cargo test --test comprehensive_integration_tests test_concurrent_access_stress_test --release"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test 8: 1ms Target Tests
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if run_test_suite "1ms Target Tests" "cargo test --test comprehensive_integration_tests test_1ms_query_time_targets_comprehensive --release"; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

echo ""
echo -e "${BLUE}📈 Running Performance Benchmarks...${NC}"

# Run benchmarks (don't count as pass/fail)
BENCHMARK_LOG="$RESULTS_DIR/benchmarks_${TIMESTAMP}.log"
if cargo bench --bench performance_benchmarks > "$BENCHMARK_LOG" 2>&1; then
    echo -e "${GREEN}✅ Benchmarks completed successfully${NC}"
    echo "   Results: $BENCHMARK_LOG"
    echo "   HTML Report: target/criterion/report/index.html"
else
    echo -e "${YELLOW}⚠️  Benchmarks completed with warnings${NC}"
    echo "   Log: $BENCHMARK_LOG"
fi

echo ""
echo -e "${BLUE}📋 Generating Performance Report...${NC}"

# Generate comprehensive report
REPORT_FILE="$RESULTS_DIR/performance_report_${TIMESTAMP}.md"

cat > "$REPORT_FILE" << EOF
# LDC Engine Performance Test Report

**Generated:** $(date)  
**System:** $(hostname)  
**CPU Cores:** $(nproc)  
**Memory:** $(free -h | grep '^Mem:' | awk '{print $2}')  
**Rust Version:** $(rustc --version)

## Test Summary

- **Total Test Suites:** $TOTAL_TESTS
- **Passed:** $PASSED_TESTS
- **Failed:** $FAILED_TESTS
- **Success Rate:** $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%

## Test Results

### ✅ Passed Tests
EOF

# Add passed tests to report
for log_file in "$RESULTS_DIR"/*_"$TIMESTAMP".log; do
    if [ -f "$log_file" ]; then
        test_name=$(basename "$log_file" "_${TIMESTAMP}.log")
        if grep -q "test result: ok" "$log_file" 2>/dev/null; then
            echo "- $test_name" >> "$REPORT_FILE"
        fi
    fi
done

cat >> "$REPORT_FILE" << EOF

### ❌ Failed Tests
EOF

# Add failed tests to report
for log_file in "$RESULTS_DIR"/*_"$TIMESTAMP".log; do
    if [ -f "$log_file" ]; then
        test_name=$(basename "$log_file" "_${TIMESTAMP}.log")
        if grep -q "test result: FAILED" "$log_file" 2>/dev/null; then
            echo "- $test_name" >> "$REPORT_FILE"
        fi
    fi
done

cat >> "$REPORT_FILE" << EOF

## Performance Highlights

### Query Time Performance
EOF

# Extract performance metrics from logs
if [ -f "$RESULTS_DIR/Performance_Validation_${TIMESTAMP}.log" ]; then
    echo "#### Workload Performance" >> "$REPORT_FILE"
    grep -E "(small_workload|medium_workload|large_workload)" "$RESULTS_DIR/Performance_Validation_${TIMESTAMP}.log" | while read -r line; do
        echo "- $line" >> "$REPORT_FILE"
    done
    
    echo "" >> "$REPORT_FILE"
    echo "#### Throughput Performance" >> "$REPORT_FILE"
    grep -E "Throughput:" "$RESULTS_DIR/Performance_Validation_${TIMESTAMP}.log" | while read -r line; do
        echo "- $line" >> "$REPORT_FILE"
    done
fi

cat >> "$REPORT_FILE" << EOF

### SIMD Optimization
EOF

if [ -f "$RESULTS_DIR/SIMD_Compatibility_${TIMESTAMP}.log" ]; then
    grep -E "(Max difference|SIMD errors)" "$RESULTS_DIR/SIMD_Compatibility_${TIMESTAMP}.log" | while read -r line; do
        echo "- $line" >> "$REPORT_FILE"
    done
fi

cat >> "$REPORT_FILE" << EOF

### Memory Efficiency
EOF

if [ -f "$RESULTS_DIR/Memory_Tests_${TIMESTAMP}.log" ]; then
    grep -E "(Pool utilization|Memory mapping|Data integrity)" "$RESULTS_DIR/Memory_Tests_${TIMESTAMP}.log" | while read -r line; do
        echo "- $line" >> "$REPORT_FILE"
    done
fi

cat >> "$REPORT_FILE" << EOF

## Detailed Logs

All detailed test logs are available in the \`$RESULTS_DIR\` directory:

EOF

# List all log files
for log_file in "$RESULTS_DIR"/*_"$TIMESTAMP".log; do
    if [ -f "$log_file" ]; then
        echo "- $(basename "$log_file")" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << EOF

## Benchmark Results

Interactive benchmark results are available at: \`target/criterion/report/index.html\`

## Recommendations

EOF

# Add recommendations based on test results
if [ $FAILED_TESTS -eq 0 ]; then
    cat >> "$REPORT_FILE" << EOF
🎉 **All tests passed!** The LDC engine performance optimizations are working correctly.

### Next Steps:
- Monitor performance in production environments
- Consider enabling all optimizations for maximum performance
- Regular performance regression testing recommended
EOF
else
    cat >> "$REPORT_FILE" << EOF
⚠️ **Some tests failed.** Review the failed test logs for details.

### Recommended Actions:
- Check failed test logs for specific issues
- Verify system requirements and dependencies
- Consider adjusting performance parameters
- Re-run tests after addressing issues
EOF
fi

cat >> "$REPORT_FILE" << EOF

---
*Report generated by LDC Engine Performance Testing Framework*
EOF

echo ""
echo -e "${BLUE}📊 Test Results Summary:${NC}"
echo -e "  Total Test Suites: $TOTAL_TESTS"
echo -e "  Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "  Failed: ${RED}$FAILED_TESTS${NC}"
echo -e "  Success Rate: $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%"
echo ""
echo -e "${BLUE}📄 Detailed Report: ${YELLOW}$REPORT_FILE${NC}"
echo -e "${BLUE}📁 All Logs: ${YELLOW}$RESULTS_DIR/${NC}"
echo ""

# Final status
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}🎉 All performance tests completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some tests failed. Check the logs for details.${NC}"
    exit 1
fi