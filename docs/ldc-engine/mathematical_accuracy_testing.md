# Mathematical Accuracy Testing Framework

## Overview

The Mathematical Accuracy Testing Framework provides comprehensive validation of Lorentzian distance calculations across different implementations (standard, SIMD, HNSW) to ensure mathematical accuracy and consistency in the LDC trading system.

## Features

### Test Categories

1. **Standard Tests**: Normal trading indicator ranges (RSI 0-100, WT -50 to 50, etc.)
2. **Edge Cases**: Zero values, negative values, mixed signs
3. **Extreme Values**: Very large/small values, float limits
4. **Precision Tests**: Floating-point precision boundaries

### Test Types

1. **Mathematical Accuracy**: Validates calculations against reference implementations
2. **SIMD Compatibility**: Ensures SIMD optimizations produce identical results
3. **HNSW Compatibility**: Verifies HNSW distance function matches exact calculations

## Usage

### Running Tests

```bash
# Run all mathematical accuracy tests
cargo test --test mathematical_accuracy_tests

# Run with output
cargo test --test mathematical_accuracy_tests -- --nocapture

# Run specific test
cargo test test_simd_vs_standard_accuracy
```

### Using the Framework in Code

```rust
use ldc_engine::*;

// Create test suite with default tolerance (1e-5)
let test_suite = MathematicalTestSuite::new();

// Or with custom tolerance
let test_suite = MathematicalTestSuite::with_tolerance(1e-6);

// Test SIMD vs standard accuracy
let simd_result = test_suite.test_simd_accuracy();
simd_result.print_detailed_results();
assert!(simd_result.all_passed());

// Test HNSW compatibility
let hnsw_result = test_suite.test_hnsw_compatibility();
hnsw_result.print_detailed_results();
assert!(hnsw_result.all_passed());

// Test mathematical accuracy
let accuracy_result = test_suite.test_mathematical_accuracy();
accuracy_result.print_detailed_results();
assert!(accuracy_result.success_rate >= 90.0);
```

### Example Output

```
=== Mathematical Accuracy Test Results ===
Total Tests: 15
Passed: 15 (100.0%)
Failed: 0

--- Category Summary ---
Standard: 4/4 passed (100.0%), max error: 1.36e-6, avg error: 6.60e-7
EdgeCases: 4/4 passed (100.0%), max error: 1.61e-6, avg error: 4.70e-7
ExtremeValues: 4/4 passed (100.0%), max error: 2.63e-6, avg error: 1.48e-6
Precision: 3/3 passed (100.0%), max error: 9.60e-8, avg error: 3.20e-8
```

## Test Cases

### Standard Cases
- Identical features (expected distance: 0.0)
- Typical RSI values (70.0 vs 30.0)
- Overbought/oversold conditions
- Neutral market conditions

### Edge Cases
- Zero features
- One zero, one non-zero
- Negative values
- Mixed positive/negative signs

### Extreme Cases
- Very large values (1e6)
- Very small values (1e-6)
- Float maximum values
- Float minimum values

### Precision Cases
- Epsilon differences
- Near-zero differences
- Precision boundaries

## Configuration

### Tolerance Levels
- Default: `1e-5` (suitable for most applications)
- Strict: `1e-6` or lower (for high-precision requirements)
- Relaxed: `1e-4` (for performance-optimized scenarios)

### Success Rate Thresholds
- Mathematical Accuracy: ≥90% success rate
- SIMD Compatibility: 100% success rate (must be identical)
- HNSW Compatibility: 100% success rate (must be identical)
- Edge Cases: ≥75% success rate
- Precision Tests: ≥60% success rate with strict tolerance

## Integration with CI/CD

The framework is designed for automated testing:

```bash
# In CI pipeline
cargo test --test mathematical_accuracy_tests
if [ $? -ne 0 ]; then
    echo "Mathematical accuracy tests failed"
    exit 1
fi
```

## Requirements Validation

The framework validates the following requirements:

- **Requirement 1.1**: Exact mathematical accuracy against reference implementations
- **Requirement 1.2**: SIMD vs standard calculations identical within floating-point precision (1e-6)
- **Requirement 1.3**: HNSW distance calculations compatible with exact Lorentzian distance formula
- **Requirement 1.4**: Detailed error analysis with input values for different results
- **Requirement 1.5**: Edge cases including zero values, NaN, infinity, and extreme ranges

## Performance Considerations

- Tests are designed to be fast and suitable for CI/CD
- SIMD tests may show slower performance on small datasets (expected behavior)
- Batch operations demonstrate SIMD benefits on larger datasets
- Memory usage is minimal for test execution

## Troubleshooting

### Common Issues

1. **Tolerance Too Strict**: Adjust tolerance if legitimate floating-point precision differences cause failures
2. **SIMD Failures**: Check CPU support for SIMD instructions
3. **Extreme Value Failures**: May indicate overflow/underflow issues in calculations

### Debug Output

Use `--nocapture` flag to see detailed test output including:
- Individual test results
- Error magnitudes
- Category summaries
- Failed test details

## Demo

Run the demo to see the framework in action:

```bash
cargo run --example mathematical_accuracy_demo
```

This demonstrates:
- Basic distance calculations
- SIMD vs standard comparison
- HNSW compatibility
- Edge case handling
- Performance comparison
- Batch operations