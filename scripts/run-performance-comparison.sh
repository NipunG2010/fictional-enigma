#!/bin/bash

# Performance Comparison Script
# Compares performance between two git branches or commits

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$PROJECT_ROOT/rust"
TEST_DIR="$RUST_DIR/end-to-end-tests"

# Default values
BASELINE_REF="main"
CURRENT_REF="HEAD"
OUTPUT_DIR="performance-comparison"
TEST_CONFIG="performance"
CLEANUP=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Compare performance between two git references"
    echo ""
    echo "OPTIONS:"
    echo "  -b, --baseline REF    Baseline git reference (default: main)"
    echo "  -c, --current REF     Current git reference (default: HEAD)"
    echo "  -o, --output DIR      Output directory (default: performance-comparison)"
    echo "  -t, --config TYPE     Test configuration type (ci|local|performance, default: performance)"
    echo "  --no-cleanup          Don't cleanup temporary directories"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Compare HEAD against main"
    echo "  $0 -b v1.0.0 -c feature-branch      # Compare feature-branch against v1.0.0"
    echo "  $0 -t ci                             # Use CI test configuration"
}

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')]${NC} $1"
}

cleanup() {
    if [ "$CLEANUP" = true ]; then
        log "Cleaning up temporary directories..."
        rm -rf "$OUTPUT_DIR/baseline-workspace" "$OUTPUT_DIR/current-workspace" 2>/dev/null || true
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--baseline)
            BASELINE_REF="$2"
            shift 2
            ;;
        -c|--current)
            CURRENT_REF="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--config)
            TEST_CONFIG="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate git references
if ! git rev-parse --verify "$BASELINE_REF" >/dev/null 2>&1; then
    log_error "Invalid baseline reference: $BASELINE_REF"
    exit 1
fi

if ! git rev-parse --verify "$CURRENT_REF" >/dev/null 2>&1; then
    log_error "Invalid current reference: $CURRENT_REF"
    exit 1
fi

# Validate test configuration
if [[ ! "$TEST_CONFIG" =~ ^(ci|local|performance)$ ]]; then
    log_error "Invalid test configuration: $TEST_CONFIG. Must be one of: ci, local, performance"
    exit 1
fi

# Setup cleanup trap
trap cleanup EXIT

log "Starting performance comparison"
log "  Baseline: $BASELINE_REF"
log "  Current:  $CURRENT_REF"
log "  Config:   $TEST_CONFIG"
log "  Output:   $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Get current branch to restore later
ORIGINAL_BRANCH=$(git -C "$PROJECT_ROOT" branch --show-current 2>/dev/null || echo "")

# Function to run tests for a specific reference
run_tests_for_ref() {
    local ref=$1
    local workspace_dir=$2
    local results_dir=$3
    
    log "Setting up workspace for $ref..."
    
    # Create workspace directory
    mkdir -p "$workspace_dir"
    
    # Copy project to workspace (to avoid affecting working directory)
    rsync -a --exclude=target --exclude=.git "$PROJECT_ROOT/" "$workspace_dir/"
    
    # Checkout the specific reference in the workspace
    (
        cd "$workspace_dir"
        git checkout "$ref" >/dev/null 2>&1
    )
    
    log "Building test framework for $ref..."
    (
        cd "$workspace_dir/rust"
        cargo build --release --package end-to-end-tests --bin test-runner --bin ci-helper >/dev/null 2>&1
    )
    
    log "Generating test configuration for $ref..."
    (
        cd "$workspace_dir/rust/end-to-end-tests"
        mkdir -p "$results_dir"
        cargo run --release --bin ci-helper -- generate-config \
            --environment "$TEST_CONFIG" \
            --output test_config.toml >/dev/null 2>&1
    )
    
    log "Running performance tests for $ref..."
    (
        cd "$workspace_dir/rust/end-to-end-tests"
        timeout 1800 cargo run --release --bin test-runner -- \
            --config test_config.toml \
            --output-dir "$results_dir" \
            --format json \
            --suite performance 2>/dev/null || {
            log_warning "Tests for $ref completed with issues (timeout or test failures)"
        }
    )
}

# Run tests for baseline
log "Running baseline tests ($BASELINE_REF)..."
run_tests_for_ref "$BASELINE_REF" "baseline-workspace" "baseline-results"

# Run tests for current
log "Running current tests ($CURRENT_REF)..."
run_tests_for_ref "$CURRENT_REF" "current-workspace" "current-results"

# Find test reports
BASELINE_REPORT=$(find baseline-results -name "test_report_*.json" | sort | tail -1)
CURRENT_REPORT=$(find current-results -name "test_report_*.json" | sort | tail -1)

if [ -z "$BASELINE_REPORT" ] || [ -z "$CURRENT_REPORT" ]; then
    log_error "Could not find test reports"
    log_error "  Baseline: $BASELINE_REPORT"
    log_error "  Current:  $CURRENT_REPORT"
    exit 1
fi

log "Analyzing performance differences..."

# Generate comparison report
(
    cd "current-workspace/rust/end-to-end-tests"
    cargo run --release --bin test-report-generator -- \
        --input "../../../$CURRENT_REPORT" \
        --output "../../../analysis" \
        --compare "../../../$BASELINE_REPORT" >/dev/null 2>&1
)

# Check for regressions
REGRESSION_EXIT_CODE=0
(
    cd "current-workspace/rust/end-to-end-tests"
    cargo run --release --bin ci-helper -- check-regressions \
        --current "../../../$CURRENT_REPORT" \
        --baseline "../../../$BASELINE_REPORT" > ../../../regression_check.log 2>&1
) || REGRESSION_EXIT_CODE=$?

# Display results
echo ""
log_success "Performance comparison completed!"
echo ""

# Extract and display key metrics
python3 -c "
import json
import sys

try:
    with open('$CURRENT_REPORT', 'r') as f:
        current = json.load(f)
    with open('$BASELINE_REPORT', 'r') as f:
        baseline = json.load(f)
    
    print('📊 Performance Comparison Results')
    print('=' * 50)
    
    # Pass rate comparison
    current_pass_rate = current['summary']['overall_pass_rate'] * 100
    baseline_pass_rate = baseline['summary']['overall_pass_rate'] * 100
    pass_rate_change = current_pass_rate - baseline_pass_rate
    
    print(f'Pass Rate:')
    print(f'  Baseline ($BASELINE_REF): {baseline_pass_rate:.1f}%')
    print(f'  Current ($CURRENT_REF):  {current_pass_rate:.1f}%')
    print(f'  Change: {pass_rate_change:+.1f}%')
    
    # Duration comparison
    current_duration = current['summary']['total_duration_minutes']
    baseline_duration = baseline['summary']['total_duration_minutes']
    duration_change = ((current_duration - baseline_duration) / baseline_duration) * 100 if baseline_duration > 0 else 0
    
    print(f'\\nTest Duration:')
    print(f'  Baseline ($BASELINE_REF): {baseline_duration:.1f} minutes')
    print(f'  Current ($CURRENT_REF):  {current_duration:.1f} minutes')
    print(f'  Change: {duration_change:+.1f}%')
    
    # Health score comparison
    current_health = current['summary']['system_health_score'] * 100
    baseline_health = baseline['summary']['system_health_score'] * 100
    health_change = current_health - baseline_health
    
    print(f'\\nSystem Health Score:')
    print(f'  Baseline ($BASELINE_REF): {baseline_health:.1f}%')
    print(f'  Current ($CURRENT_REF):  {current_health:.1f}%')
    print(f'  Change: {health_change:+.1f}%')
    
except Exception as e:
    print(f'Error processing reports: {e}', file=sys.stderr)
    sys.exit(1)
"

echo ""

# Display regression analysis
if [ $REGRESSION_EXIT_CODE -eq 0 ]; then
    log_success "✅ No significant performance regressions detected"
else
    log_warning "⚠️  Performance regressions detected:"
    echo ""
    cat regression_check.log | grep -E "(CRITICAL|HIGH|MEDIUM|LOW|regression)" || true
fi

echo ""
log "📁 Results saved to: $OUTPUT_DIR/"
log "   • Baseline results: baseline-results/"
log "   • Current results:  current-results/"
log "   • Analysis:         analysis/"
log "   • Regression check: regression_check.log"

if [ -f "analysis/comparison_report.json" ]; then
    log "   • Comparison report: analysis/comparison_report.json"
fi

if [ -f "analysis/report.html" ]; then
    log "   • HTML report:      analysis/report.html"
    log ""
    log "💡 Open analysis/report.html in your browser to view the detailed report"
fi

# Restore original branch if we were on one
if [ -n "$ORIGINAL_BRANCH" ]; then
    git -C "$PROJECT_ROOT" checkout "$ORIGINAL_BRANCH" >/dev/null 2>&1 || true
fi

exit $REGRESSION_EXIT_CODE