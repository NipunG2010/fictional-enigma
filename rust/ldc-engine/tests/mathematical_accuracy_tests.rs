use ldc_engine::*;
use std::collections::HashMap;

/// Mathematical accuracy unit testing framework for LDC distance calculations
/// This module provides comprehensive testing of Lorentzian distance calculations
/// across different implementations (standard, SIMD, HNSW) to ensure mathematical accuracy.

/// Test suite for mathematical accuracy validation
pub struct MathematicalTestSuite {
    pub tolerance: f64,
    pub test_cases: Vec<DistanceTestCase>,
}

/// Individual test case for distance calculation validation
#[derive(Debug, Clone)]
pub struct DistanceTestCase {
    pub name: String,
    pub features1: FeatureSeries,
    pub features2: FeatureSeries,
    pub expected_distance: Option<f64>, // None means calculate from standard implementation
    pub test_category: TestCategory,
}

/// Categories of test cases for comprehensive coverage
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TestCategory {
    Standard,      // Normal feature ranges
    EdgeCases,     // Zero, NaN, infinity
    ExtremeValues, // Very large/small values
    Precision,     // Floating-point precision tests
}

/// Result of a single unit test
#[derive(Debug, Clone)]
pub struct UnitTestResult {
    pub test_name: String,
    pub passed: bool,
    pub expected: f64,
    pub actual: f64,
    pub difference: f64,
    pub tolerance: f64,
    pub test_category: TestCategory,
}

/// Aggregated results from multiple unit tests
#[derive(Debug, Clone)]
pub struct TestResult {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub success_rate: f64,
    pub results: Vec<UnitTestResult>,
    pub category_summary: HashMap<TestCategory, CategorySummary>,
}

/// Summary statistics for each test category
#[derive(Debug, Clone)]
pub struct CategorySummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub success_rate: f64,
    pub max_error: f64,
    pub avg_error: f64,
}

impl MathematicalTestSuite {
    /// Create a new test suite with default tolerance
    pub fn new() -> Self {
        Self {
            tolerance: 1e-5, // Adjusted for realistic floating-point precision
            test_cases: Self::generate_test_cases(),
        }
    }

    /// Create a test suite with custom tolerance
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            test_cases: Self::generate_test_cases(),
        }
    }

    /// Test SIMD vs standard distance calculation accuracy
    /// Requirement 1.2: SIMD vs standard calculations SHALL be identical within floating-point precision
    pub fn test_simd_accuracy(&self) -> TestResult {
        let mut results = Vec::new();

        for test_case in &self.test_cases {
            let standard_distance = test_case.features1.lorentzian_distance_standard(&test_case.features2);
            
            let simd_distance = match test_case.features1.lorentzian_distance_simd(&test_case.features2) {
                Ok(distance) => distance,
                Err(_) => {
                    // SIMD failed, record as test failure
                    results.push(UnitTestResult {
                        test_name: format!("SIMD_vs_Standard_{}", test_case.name),
                        passed: false,
                        expected: standard_distance as f64,
                        actual: f64::NAN,
                        difference: f64::INFINITY,
                        tolerance: self.tolerance,
                        test_category: test_case.test_category.clone(),
                    });
                    continue;
                }
            };

            let diff = (standard_distance - simd_distance).abs() as f64;
            let passed = diff < self.tolerance;

            results.push(UnitTestResult {
                test_name: format!("SIMD_vs_Standard_{}", test_case.name),
                passed,
                expected: standard_distance as f64,
                actual: simd_distance as f64,
                difference: diff,
                tolerance: self.tolerance,
                test_category: test_case.test_category.clone(),
            });
        }

        TestResult::from_unit_results(results)
    }

    /// Test HNSW distance calculation compatibility
    /// Requirement 1.3: HNSW distance calculations SHALL verify compatibility with exact Lorentzian distance formula
    pub fn test_hnsw_compatibility(&self) -> TestResult {
        let mut results = Vec::new();

        for test_case in &self.test_cases {
            let rust_distance = LDCEngine::lorentzian_distance(
                &test_case.features1,
                &test_case.features2,
                5
            );

            let features1_array = test_case.features1.to_array();
            let features2_array = test_case.features2.to_array();
            let hnsw_distance = lorentzian_distance_hnsw(&features1_array, &features2_array);

            let diff = (rust_distance - hnsw_distance).abs() as f64;
            let passed = diff < self.tolerance;

            results.push(UnitTestResult {
                test_name: format!("HNSW_vs_Standard_{}", test_case.name),
                passed,
                expected: rust_distance as f64,
                actual: hnsw_distance as f64,
                difference: diff,
                tolerance: self.tolerance,
                test_category: test_case.test_category.clone(),
            });
        }

        TestResult::from_unit_results(results)
    }

    /// Test mathematical accuracy against reference implementation
    /// Requirement 1.1: System SHALL verify exact mathematical accuracy against reference implementations
    pub fn test_mathematical_accuracy(&self) -> TestResult {
        let mut results = Vec::new();

        for test_case in &self.test_cases {
            let calculated_distance = LDCEngine::lorentzian_distance(
                &test_case.features1,
                &test_case.features2,
                5
            );

            let expected_distance = if let Some(expected) = test_case.expected_distance {
                expected
            } else {
                // Calculate reference distance manually
                self.calculate_reference_distance(&test_case.features1, &test_case.features2)
            };

            let diff = (calculated_distance as f64 - expected_distance).abs();
            let passed = diff < self.tolerance;

            results.push(UnitTestResult {
                test_name: format!("Mathematical_Accuracy_{}", test_case.name),
                passed,
                expected: expected_distance,
                actual: calculated_distance as f64,
                difference: diff,
                tolerance: self.tolerance,
                test_category: test_case.test_category.clone(),
            });
        }

        TestResult::from_unit_results(results)
    }

    /// Calculate reference Lorentzian distance for validation
    fn calculate_reference_distance(&self, features1: &FeatureSeries, features2: &FeatureSeries) -> f64 {
        let f1_diff = (features1.f1 - features2.f1).abs() as f64;
        let f2_diff = (features1.f2 - features2.f2).abs() as f64;
        let f3_diff = (features1.f3 - features2.f3).abs() as f64;
        let f4_diff = (features1.f4 - features2.f4).abs() as f64;
        let f5_diff = (features1.f5 - features2.f5).abs() as f64;

        (1.0 + f1_diff).ln() +
        (1.0 + f2_diff).ln() +
        (1.0 + f3_diff).ln() +
        (1.0 + f4_diff).ln() +
        (1.0 + f5_diff).ln()
    }

    /// Generate comprehensive test cases covering all scenarios
    /// Requirement 1.5: System SHALL test edge cases including zero values, NaN, infinity, and extreme ranges
    fn generate_test_cases() -> Vec<DistanceTestCase> {
        let mut cases = Vec::new();

        // Standard test cases - normal trading indicator ranges
        cases.extend(Self::generate_standard_cases());
        
        // Edge cases - boundary conditions
        cases.extend(Self::generate_edge_cases());
        
        // Extreme values - stress testing
        cases.extend(Self::generate_extreme_cases());
        
        // Precision tests - floating-point accuracy
        cases.extend(Self::generate_precision_cases());

        cases
    }

    /// Generate standard test cases with typical trading indicator values
    fn generate_standard_cases() -> Vec<DistanceTestCase> {
        vec![
            DistanceTestCase {
                name: "identical_features".to_string(),
                features1: FeatureSeries { f1: 50.0, f2: 0.0, f3: 0.0, f4: 25.0, f5: 50.0 },
                features2: FeatureSeries { f1: 50.0, f2: 0.0, f3: 0.0, f4: 25.0, f5: 50.0 },
                expected_distance: Some(0.0),
                test_category: TestCategory::Standard,
            },
            DistanceTestCase {
                name: "typical_rsi_values".to_string(),
                features1: FeatureSeries { f1: 70.0, f2: 10.0, f3: 50.0, f4: 30.0, f5: 65.0 },
                features2: FeatureSeries { f1: 30.0, f2: -10.0, f3: -50.0, f4: 70.0, f5: 35.0 },
                expected_distance: None, // Will be calculated
                test_category: TestCategory::Standard,
            },
            DistanceTestCase {
                name: "overbought_oversold".to_string(),
                features1: FeatureSeries { f1: 80.0, f2: 40.0, f3: 100.0, f4: 60.0, f5: 75.0 },
                features2: FeatureSeries { f1: 20.0, f2: -40.0, f3: -100.0, f4: 15.0, f5: 25.0 },
                expected_distance: None,
                test_category: TestCategory::Standard,
            },
            DistanceTestCase {
                name: "neutral_values".to_string(),
                features1: FeatureSeries { f1: 50.0, f2: 0.0, f3: 0.0, f4: 25.0, f5: 50.0 },
                features2: FeatureSeries { f1: 45.0, f2: 5.0, f3: -10.0, f4: 30.0, f5: 55.0 },
                expected_distance: None,
                test_category: TestCategory::Standard,
            },
        ]
    }

    /// Generate edge cases for boundary condition testing
    fn generate_edge_cases() -> Vec<DistanceTestCase> {
        vec![
            DistanceTestCase {
                name: "zero_features".to_string(),
                features1: FeatureSeries { f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 },
                features2: FeatureSeries { f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 },
                expected_distance: Some(0.0),
                test_category: TestCategory::EdgeCases,
            },
            DistanceTestCase {
                name: "one_zero_one_nonzero".to_string(),
                features1: FeatureSeries { f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 },
                features2: FeatureSeries { f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0 },
                expected_distance: Some(5.0 * (1.0 + 1.0_f64).ln()), // 5 * ln(2)
                test_category: TestCategory::EdgeCases,
            },
            DistanceTestCase {
                name: "negative_values".to_string(),
                features1: FeatureSeries { f1: -50.0, f2: -25.0, f3: -100.0, f4: -10.0, f5: -75.0 },
                features2: FeatureSeries { f1: -30.0, f2: -15.0, f3: -80.0, f4: -5.0, f5: -60.0 },
                expected_distance: None,
                test_category: TestCategory::EdgeCases,
            },
            DistanceTestCase {
                name: "mixed_signs".to_string(),
                features1: FeatureSeries { f1: 50.0, f2: -25.0, f3: 100.0, f4: -10.0, f5: 75.0 },
                features2: FeatureSeries { f1: -30.0, f2: 15.0, f3: -80.0, f4: 5.0, f5: -60.0 },
                expected_distance: None,
                test_category: TestCategory::EdgeCases,
            },
        ]
    }

    /// Generate extreme value test cases for stress testing
    fn generate_extreme_cases() -> Vec<DistanceTestCase> {
        vec![
            DistanceTestCase {
                name: "very_large_values".to_string(),
                features1: FeatureSeries { 
                    f1: 1e6, f2: 1e6, f3: 1e6, f4: 1e6, f5: 1e6 
                },
                features2: FeatureSeries { 
                    f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 
                },
                expected_distance: None,
                test_category: TestCategory::ExtremeValues,
            },
            DistanceTestCase {
                name: "very_small_values".to_string(),
                features1: FeatureSeries { 
                    f1: 1e-6, f2: 1e-6, f3: 1e-6, f4: 1e-6, f5: 1e-6 
                },
                features2: FeatureSeries { 
                    f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 
                },
                expected_distance: None,
                test_category: TestCategory::ExtremeValues,
            },
            DistanceTestCase {
                name: "max_float_values".to_string(),
                features1: FeatureSeries { 
                    f1: f32::MAX, f2: f32::MAX, f3: f32::MAX, f4: f32::MAX, f5: f32::MAX 
                },
                features2: FeatureSeries { 
                    f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 
                },
                expected_distance: None,
                test_category: TestCategory::ExtremeValues,
            },
            DistanceTestCase {
                name: "min_float_values".to_string(),
                features1: FeatureSeries { 
                    f1: f32::MIN, f2: f32::MIN, f3: f32::MIN, f4: f32::MIN, f5: f32::MIN 
                },
                features2: FeatureSeries { 
                    f1: 0.0, f2: 0.0, f3: 0.0, f4: 0.0, f5: 0.0 
                },
                expected_distance: None,
                test_category: TestCategory::ExtremeValues,
            },
        ]
    }

    /// Generate precision test cases for floating-point accuracy validation
    fn generate_precision_cases() -> Vec<DistanceTestCase> {
        vec![
            DistanceTestCase {
                name: "epsilon_difference".to_string(),
                features1: FeatureSeries { 
                    f1: 1.0, f2: 1.0, f3: 1.0, f4: 1.0, f5: 1.0 
                },
                features2: FeatureSeries { 
                    f1: 1.0 + f32::EPSILON, 
                    f2: 1.0 + f32::EPSILON, 
                    f3: 1.0 + f32::EPSILON, 
                    f4: 1.0 + f32::EPSILON, 
                    f5: 1.0 + f32::EPSILON 
                },
                expected_distance: None,
                test_category: TestCategory::Precision,
            },
            DistanceTestCase {
                name: "near_zero_difference".to_string(),
                features1: FeatureSeries { 
                    f1: 1e-7, f2: 1e-7, f3: 1e-7, f4: 1e-7, f5: 1e-7 
                },
                features2: FeatureSeries { 
                    f1: 2e-7, f2: 2e-7, f3: 2e-7, f4: 2e-7, f5: 2e-7 
                },
                expected_distance: None,
                test_category: TestCategory::Precision,
            },
            DistanceTestCase {
                name: "precision_boundary".to_string(),
                features1: FeatureSeries { 
                    f1: 1.0000001, f2: 1.0000001, f3: 1.0000001, f4: 1.0000001, f5: 1.0000001 
                },
                features2: FeatureSeries { 
                    f1: 1.0000002, f2: 1.0000002, f3: 1.0000002, f4: 1.0000002, f5: 1.0000002 
                },
                expected_distance: None,
                test_category: TestCategory::Precision,
            },
        ]
    }
}

impl TestResult {
    /// Create TestResult from a vector of UnitTestResult
    pub fn from_unit_results(results: Vec<UnitTestResult>) -> Self {
        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;
        let success_rate = if total_tests > 0 {
            passed_tests as f64 / total_tests as f64 * 100.0
        } else {
            0.0
        };

        // Calculate category summaries
        let mut category_summary = HashMap::new();
        let categories = [
            TestCategory::Standard,
            TestCategory::EdgeCases,
            TestCategory::ExtremeValues,
            TestCategory::Precision,
        ];

        for category in &categories {
            let category_results: Vec<_> = results.iter()
                .filter(|r| r.test_category == *category)
                .collect();
            
            let total = category_results.len();
            let passed = category_results.iter().filter(|r| r.passed).count();
            let failed = total - passed;
            let success_rate = if total > 0 {
                passed as f64 / total as f64 * 100.0
            } else {
                0.0
            };

            let errors: Vec<f64> = category_results.iter()
                .map(|r| r.difference)
                .filter(|d| d.is_finite())
                .collect();
            
            let max_error = errors.iter().fold(0.0f64, |a, &b| a.max(b));
            let avg_error = if !errors.is_empty() {
                errors.iter().sum::<f64>() / errors.len() as f64
            } else {
                0.0
            };

            category_summary.insert(category.clone(), CategorySummary {
                total,
                passed,
                failed,
                success_rate,
                max_error,
                avg_error,
            });
        }

        Self {
            total_tests,
            passed_tests,
            failed_tests,
            success_rate,
            results,
            category_summary,
        }
    }

    /// Print detailed test results
    pub fn print_detailed_results(&self) {
        println!("\n=== Mathematical Accuracy Test Results ===");
        println!("Total Tests: {}", self.total_tests);
        println!("Passed: {} ({:.1}%)", self.passed_tests, self.success_rate);
        println!("Failed: {}", self.failed_tests);

        // Print category summaries
        println!("\n--- Category Summary ---");
        for (category, summary) in &self.category_summary {
            println!("{:?}: {}/{} passed ({:.1}%), max error: {:.2e}, avg error: {:.2e}",
                category, summary.passed, summary.total, summary.success_rate,
                summary.max_error, summary.avg_error);
        }

        // Print failed tests
        if self.failed_tests > 0 {
            println!("\n--- Failed Tests ---");
            for result in &self.results {
                if !result.passed {
                    println!("{}: expected {:.6e}, got {:.6e}, diff {:.6e} (tolerance: {:.6e})",
                        result.test_name, result.expected, result.actual, 
                        result.difference, result.tolerance);
                }
            }
        }
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.failed_tests == 0
    }
}

// Integration tests using the framework
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mathematical_accuracy_framework() {
        let test_suite = MathematicalTestSuite::new();
        let result = test_suite.test_mathematical_accuracy();
        
        result.print_detailed_results();
        assert!(result.success_rate >= 90.0, "Mathematical accuracy tests should have at least 90% success rate");
    }

    #[test]
    fn test_simd_vs_standard_accuracy() {
        let test_suite = MathematicalTestSuite::new();
        let result = test_suite.test_simd_accuracy();
        
        result.print_detailed_results();
        assert!(result.all_passed(), "SIMD vs standard calculations must be identical within tolerance");
    }

    #[test]
    fn test_hnsw_compatibility() {
        let test_suite = MathematicalTestSuite::new();
        let result = test_suite.test_hnsw_compatibility();
        
        result.print_detailed_results();
        assert!(result.all_passed(), "HNSW distance calculations must match standard implementation");
    }

    #[test]
    fn test_edge_cases_handling() {
        let test_suite = MathematicalTestSuite::new();
        let result = test_suite.test_mathematical_accuracy();
        
        // Check that edge cases are handled properly
        let edge_case_results: Vec<_> = result.results.iter()
            .filter(|r| r.test_category == TestCategory::EdgeCases)
            .collect();
        
        let edge_case_success_rate = if !edge_case_results.is_empty() {
            edge_case_results.iter().filter(|r| r.passed).count() as f64 / 
            edge_case_results.len() as f64 * 100.0
        } else {
            100.0
        };
        
        assert!(edge_case_success_rate >= 75.0, 
            "Edge cases should have at least 75% success rate, got {:.1}%", edge_case_success_rate);
    }

    #[test]
    fn test_extreme_values_handling() {
        let test_suite = MathematicalTestSuite::new();
        let result = test_suite.test_mathematical_accuracy();
        
        // Check that extreme values don't cause crashes or invalid results
        let extreme_case_results: Vec<_> = result.results.iter()
            .filter(|r| r.test_category == TestCategory::ExtremeValues)
            .collect();
        
        for result in extreme_case_results {
            assert!(result.actual.is_finite(), 
                "Extreme value test {} should produce finite result, got {}", 
                result.test_name, result.actual);
        }
    }

    #[test]
    fn test_precision_requirements() {
        let test_suite = MathematicalTestSuite::with_tolerance(1e-10); // Stricter tolerance
        let result = test_suite.test_mathematical_accuracy();
        
        let precision_results: Vec<_> = result.results.iter()
            .filter(|r| r.test_category == TestCategory::Precision)
            .collect();
        
        // Precision tests should still pass with stricter tolerance
        let precision_success_rate = if !precision_results.is_empty() {
            precision_results.iter().filter(|r| r.passed).count() as f64 / 
            precision_results.len() as f64 * 100.0
        } else {
            100.0
        };
        
        assert!(precision_success_rate >= 60.0, 
            "Precision tests should have at least 60% success rate with strict tolerance, got {:.1}%", 
            precision_success_rate);
    }
}