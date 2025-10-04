#!/bin/bash

# CI/CD Test Runner Script for LDC Engine
# This script provides automated test execution with proper error handling,
# timeout management, and CI/CD integration support.

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_REPORTS_DIR="${PROJECT_ROOT}/test_reports"
CONFIG_FILE="${PROJECT_ROOT}/test_runner_config.json"
LOG_FILE="${TEST_REPORTS_DIR}/ci_test_runner.log"

# Default values
PARALLEL_JOBS=${PARALLEL_JOBS:-$(nproc)}
TIMEOUT=${TIMEOUT:-300}
VERBOSE=${VERBOSE:-false}
CATEGORIES=${CATEGORIES:-""}
CHANGED_FILES=${CHANGED_FILES:-""}
PATTERN=${PATTERN:-""}
REGRESSION_THRESHOLD=${REGRESSION_THRESHOLD:-10.0}
CLEANUP_TIMEOUT=${CLEANUP_TIMEOUT:-30}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    log_info "Performing cleanup..."
    
    # Kill any remaining test processes
    pkill -f "cargo test" || true
    pkill -f "test_runner" || true
    
    # Clean up temporary files
    find "$TEST_REPORTS_DIR" -name "*.tmp" -delete 2>/dev/null || true
    
    # Archive old reports if successful
    if [ $exit_code -eq 0 ]; then
        archive_old_reports
    fi
    
    log_info "Cleanup completed"
    exit $exit_code
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if cargo is available
    if ! command -v cargo &> /dev/null; then
        log_error "cargo is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we're in a Rust project
    if [ ! -f "$PROJECT_ROOT/Cargo.toml" ]; then
        log_error "Not in a Rust project directory (Cargo.toml not found)"
        exit 1
    fi
    
    # Create test reports directory
    mkdir -p "$TEST_REPORTS_DIR"
    
    # Initialize log file
    echo "CI Test Runner Log - $(date)" > "$LOG_FILE"
    
    log_success "Prerequisites check passed"
}

# Function to detect changed files from git
detect_changed_files() {
    if [ -n "$CHANGED_FILES" ]; then
        echo "$CHANGED_FILES"
        return
    fi
    
    # Try to detect changed files from git
    if git rev-parse --git-dir > /dev/null 2>&1; then
        # Get changed files from last commit or merge base
        local base_ref="${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-main}"
        if git rev-parse --verify "$base_ref" > /dev/null 2>&1; then
            git diff --name-only "$base_ref"...HEAD
        else
            git diff --name-only HEAD~1
        fi
    fi
}

# Function to build the test runner
build_test_runner() {
    log_info "Building test runner..."
    
    cd "$PROJECT_ROOT"
    
    # Build in release mode for better performance
    if ! cargo build --release --bin test_runner; then
        log_error "Failed to build test runner"
        exit 1
    fi
    
    log_success "Test runner built successfully"
}

# Function to run tests with timeout and monitoring
run_tests() {
    log_info "Starting test execution..."
    
    local cmd_args=()
    cmd_args+=("--output-dir" "$TEST_REPORTS_DIR")
    cmd_args+=("--parallel" "$PARALLEL_JOBS")
    cmd_args+=("--timeout" "$TIMEOUT")
    cmd_args+=("--regression-threshold" "$REGRESSION_THRESHOLD")
    cmd_args+=("--cleanup-timeout" "$CLEANUP_TIMEOUT")
    cmd_args+=("--machine-readable")
    
    if [ "$VERBOSE" = "true" ]; then
        cmd_args+=("--verbose")
    fi
    
    if [ -f "$CONFIG_FILE" ]; then
        cmd_args+=("--config" "$CONFIG_FILE")
    fi
    
    # Add test selection based on environment
    if [ -n "$CATEGORIES" ]; then
        cmd_args+=("--categories" "$CATEGORIES")
    elif [ -n "$PATTERN" ]; then
        cmd_args+=("--pattern" "$PATTERN")
    else
        # Try to detect changed files for smart test selection
        local changed_files
        changed_files=$(detect_changed_files)
        if [ -n "$changed_files" ]; then
            log_info "Detected changed files, running affected tests"
            # Convert newlines to commas for the command line
            local files_csv
            files_csv=$(echo "$changed_files" | tr '\n' ',' | sed 's/,$//')
            if [ -n "$files_csv" ]; then
                cmd_args+=("--changed-files" "$files_csv")
            fi
        fi
    fi
    
    # Run the test runner with timeout
    local test_runner_path="$PROJECT_ROOT/target/release/test_runner"
    
    log_info "Executing: $test_runner_path ${cmd_args[*]}"
    
    # Use timeout command if available
    if command -v timeout &> /dev/null; then
        local total_timeout=$((TIMEOUT + 60)) # Add buffer for cleanup
        if ! timeout "$total_timeout" "$test_runner_path" "${cmd_args[@]}"; then
            local exit_code=$?
            if [ $exit_code -eq 124 ]; then
                log_error "Test execution timed out after ${total_timeout} seconds"
            else
                log_error "Test execution failed with exit code $exit_code"
            fi
            return $exit_code
        fi
    else
        if ! "$test_runner_path" "${cmd_args[@]}"; then
            local exit_code=$?
            log_error "Test execution failed with exit code $exit_code"
            return $exit_code
        fi
    fi
    
    log_success "Test execution completed successfully"
}

# Function to process test results
process_results() {
    log_info "Processing test results..."
    
    # Check if reports were generated
    local latest_json="$TEST_REPORTS_DIR/latest.json"
    local latest_xml="$TEST_REPORTS_DIR/latest.xml"
    local latest_html="$TEST_REPORTS_DIR/latest.html"
    
    if [ ! -f "$latest_json" ]; then
        log_warn "No JSON test report found"
    else
        log_info "JSON report: $latest_json"
        
        # Extract key metrics for CI/CD
        if command -v jq &> /dev/null; then
            local total_suites
            local passed_suites
            local failed_suites
            local success_rate
            
            total_suites=$(jq -r '.summary.total_suites' "$latest_json")
            passed_suites=$(jq -r '.summary.passed_suites' "$latest_json")
            failed_suites=$(jq -r '.summary.failed_suites' "$latest_json")
            success_rate=$(jq -r '.summary.success_rate' "$latest_json")
            
            log_info "Test Summary: $passed_suites/$total_suites passed (${success_rate}%)"
            
            if [ "$failed_suites" -gt 0 ]; then
                log_error "$failed_suites test suite(s) failed"
                
                # Extract failed test names
                local failed_names
                failed_names=$(jq -r '.results[] | select(.status == "Failed") | .suite_name' "$latest_json" | tr '\n' ' ')
                log_error "Failed suites: $failed_names"
            fi
            
            # Check for performance regressions
            local regressions_count
            regressions_count=$(jq -r '.performance_regressions | length' "$latest_json")
            if [ "$regressions_count" -gt 0 ]; then
                log_warn "$regressions_count performance regression(s) detected"
                
                # Extract critical regressions
                local critical_regressions
                critical_regressions=$(jq -r '.performance_regressions[] | select(.severity == "Critical") | .suite_name' "$latest_json" | tr '\n' ' ')
                if [ -n "$critical_regressions" ]; then
                    log_error "Critical performance regressions in: $critical_regressions"
                fi
            fi
        fi
    fi
    
    if [ -f "$latest_xml" ]; then
        log_info "JUnit XML report: $latest_xml"
    fi
    
    if [ -f "$latest_html" ]; then
        log_info "HTML report: $latest_html"
    fi
}

# Function to archive old reports
archive_old_reports() {
    log_info "Archiving old test reports..."
    
    local archive_dir="$TEST_REPORTS_DIR/archive"
    mkdir -p "$archive_dir"
    
    # Move reports older than 7 days to archive
    find "$TEST_REPORTS_DIR" -maxdepth 1 -name "test_run_*.json" -mtime +7 -exec mv {} "$archive_dir/" \; 2>/dev/null || true
    find "$TEST_REPORTS_DIR" -maxdepth 1 -name "test_run_*.xml" -mtime +7 -exec mv {} "$archive_dir/" \; 2>/dev/null || true
    find "$TEST_REPORTS_DIR" -maxdepth 1 -name "test_run_*.html" -mtime +7 -exec mv {} "$archive_dir/" \; 2>/dev/null || true
    
    # Keep only last 20 archived reports
    local archived_reports
    archived_reports=$(find "$archive_dir" -name "test_run_*.json" | wc -l)
    if [ "$archived_reports" -gt 20 ]; then
        find "$archive_dir" -name "test_run_*.json" -printf '%T@ %p\n' | sort -n | head -n -20 | cut -d' ' -f2- | xargs rm -f
        find "$archive_dir" -name "test_run_*.xml" -printf '%T@ %p\n' | sort -n | head -n -20 | cut -d' ' -f2- | xargs rm -f
        find "$archive_dir" -name "test_run_*.html" -printf '%T@ %p\n' | sort -n | head -n -20 | cut -d' ' -f2- | xargs rm -f
    fi
}

# Function to print usage
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

CI/CD Test Runner for LDC Engine

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    -j, --jobs NUM          Number of parallel jobs (default: number of CPU cores)
    -t, --timeout SECONDS   Timeout for test suites (default: 300)
    -c, --categories LIST   Test categories to run (comma-separated)
    -f, --changed-files LIST Changed files (comma-separated)
    -p, --pattern PATTERN   Test pattern to match
    --regression-threshold PERCENT Performance regression threshold (default: 10.0)
    --cleanup-timeout SECONDS Cleanup timeout (default: 30)

ENVIRONMENT VARIABLES:
    PARALLEL_JOBS           Number of parallel jobs
    TIMEOUT                 Test timeout in seconds
    VERBOSE                 Enable verbose output (true/false)
    CATEGORIES              Test categories to run
    CHANGED_FILES           Changed files list
    PATTERN                 Test pattern
    REGRESSION_THRESHOLD    Performance regression threshold
    CLEANUP_TIMEOUT         Cleanup timeout

EXAMPLES:
    # Run all tests
    $0

    # Run with verbose output and 8 parallel jobs
    $0 --verbose --jobs 8

    # Run only unit and integration tests
    $0 --categories unit,integration

    # Run tests affected by specific files
    $0 --changed-files src/lib.rs,src/automated_test_runner.rs

    # Run tests matching a pattern
    $0 --pattern performance

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -c|--categories)
            CATEGORIES="$2"
            shift 2
            ;;
        -f|--changed-files)
            CHANGED_FILES="$2"
            shift 2
            ;;
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        --regression-threshold)
            REGRESSION_THRESHOLD="$2"
            shift 2
            ;;
        --cleanup-timeout)
            CLEANUP_TIMEOUT="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "Starting CI/CD test runner for LDC Engine"
    log_info "Configuration: parallel_jobs=$PARALLEL_JOBS, timeout=$TIMEOUT, verbose=$VERBOSE"
    
    check_prerequisites
    build_test_runner
    run_tests
    process_results
    
    log_success "CI/CD test runner completed successfully"
}

# Run main function
main "$@"