#!/usr/bin/env python3
"""
Comprehensive test runner for HMM research environment.

This script runs all research environment tests and generates a detailed report.
"""

import sys
import subprocess
import argparse
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Any


class ResearchTestRunner:
    """
    Comprehensive test runner for research environment.
    
    Runs tests in categories:
    - Unit tests
    - Integration tests
    - Performance benchmarks
    - Notebook tests
    - Compatibility tests
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize test runner."""
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_suite(
        self,
        test_file: str,
        markers: List[str] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Run a test suite and capture results.
        
        Args:
            test_file: Path to test file
            markers: Pytest markers to filter tests
            timeout: Maximum execution time
            
        Returns:
            Dictionary with test results
        """
        cmd = ['pytest', test_file, '-v', '--tb=short']
        
        if markers:
            marker_expr = ' and '.join(markers)
            cmd.extend(['-m', marker_expr])
        
        if self.verbose:
            cmd.append('-s')
        
        # Add JSON report
        report_file = f"test_report_{Path(test_file).stem}.json"
        cmd.extend(['--json-report', f'--json-report-file={report_file}'])
        
        print(f"\n{'='*60}")
        print(f"Running: {test_file}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            # Parse results
            test_results = {
                'test_file': test_file,
                'return_code': result.returncode,
                'execution_time': execution_time,
                'success': result.returncode == 0,
                'stdout': result.stdout if self.verbose else '',
                'stderr': result.stderr if result.returncode != 0 else ''
            }
            
            # Try to load JSON report
            if Path(report_file).exists():
                with open(report_file, 'r') as f:
                    json_report = json.load(f)
                    test_results['summary'] = json_report.get('summary', {})
                Path(report_file).unlink()  # Clean up
            
            return test_results
            
        except subprocess.TimeoutExpired:
            return {
                'test_file': test_file,
                'return_code': -1,
                'success': False,
                'error': f'Test suite timed out after {timeout} seconds'
            }
        except Exception as e:
            return {
                'test_file': test_file,
                'return_code': -1,
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(
        self,
        include_slow: bool = False,
        include_notebooks: bool = False
    ) -> Dict[str, Any]:
        """
        Run all research environment tests.
        
        Args:
            include_slow: Include slow-running tests
            include_notebooks: Include notebook execution tests
            
        Returns:
            Dictionary with all test results
        """
        self.start_time = time.time()
        
        test_suites = [
            {
                'name': 'Core Research Framework',
                'file': 'test_research_framework.py',
                'markers': [] if include_slow else ['not slow']
            },
            {
                'name': 'Notebook Integration',
                'file': 'test_notebook_integration.py',
                'markers': [] if include_notebooks else ['not slow']
            },
            {
                'name': 'Performance & Compatibility',
                'file': 'test_performance_compatibility.py',
                'markers': [] if include_slow else ['not slow']
            },
            {
                'name': 'HMM Core',
                'file': 'test_hmm.py',
                'markers': [] if include_slow else ['not slow']
            },
            {
                'name': 'Evaluation Framework',
                'file': 'test_evaluation.py',
                'markers': []
            },
            {
                'name': 'Visualization',
                'file': 'test_visualization.py',
                'markers': []
            },
            {
                'name': 'Data Integration',
                'file': 'test_data_integration.py',
                'markers': []
            },
            {
                'name': 'Artifact Management',
                'file': 'test_artifact_management.py',
                'markers': []
            },
            {
                'name': 'Regime Analysis',
                'file': 'test_regime_analysis.py',
                'markers': []
            }
        ]
        
        for suite in test_suites:
            test_file = Path(__file__).parent / suite['file']
            
            if not test_file.exists():
                print(f"Skipping {suite['name']}: file not found")
                continue
            
            result = self.run_test_suite(
                str(test_file),
                markers=suite['markers']
            )
            
            self.results[suite['name']] = result
        
        self.end_time = time.time()
        return self.results
    
    def generate_report(self) -> str:
        """
        Generate comprehensive test report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No test results available"
        
        total_time = self.end_time - self.start_time if self.end_time else 0
        
        report = [
            "=" * 80,
            "HMM RESEARCH ENVIRONMENT TEST REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Execution Time: {total_time:.2f} seconds",
            "",
            "SUMMARY",
            "-" * 80
        ]
        
        total_suites = len(self.results)
        passed_suites = sum(1 for r in self.results.values() if r['success'])
        failed_suites = total_suites - passed_suites
        
        report.extend([
            f"Total Test Suites: {total_suites}",
            f"Passed: {passed_suites}",
            f"Failed: {failed_suites}",
            f"Success Rate: {(passed_suites/total_suites*100):.1f}%",
            "",
            "DETAILED RESULTS",
            "-" * 80
        ])
        
        for suite_name, result in self.results.items():
            status = "✓ PASSED" if result['success'] else "✗ FAILED"
            report.append(f"\n{suite_name}: {status}")
            report.append(f"  Execution Time: {result.get('execution_time', 0):.2f}s")
            
            if 'summary' in result:
                summary = result['summary']
                report.append(f"  Tests: {summary.get('total', 0)}")
                report.append(f"  Passed: {summary.get('passed', 0)}")
                report.append(f"  Failed: {summary.get('failed', 0)}")
                report.append(f"  Skipped: {summary.get('skipped', 0)}")
            
            if not result['success'] and 'error' in result:
                report.append(f"  Error: {result['error']}")
            
            if not result['success'] and result.get('stderr'):
                report.append(f"  stderr: {result['stderr'][:200]}...")
        
        report.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(report)
    
    def save_report(self, output_file: str = "research_test_report.txt"):
        """
        Save test report to file.
        
        Args:
            output_file: Path to output file
        """
        report = self.generate_report()
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {output_file}")
    
    def save_json_results(self, output_file: str = "research_test_results.json"):
        """
        Save test results as JSON.
        
        Args:
            output_file: Path to output file
        """
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_time': self.end_time - self.start_time if self.end_time else 0,
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"JSON results saved to: {output_file}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive tests for HMM research environment'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--include-slow',
        action='store_true',
        help='Include slow-running tests'
    )
    
    parser.add_argument(
        '--include-notebooks',
        action='store_true',
        help='Include notebook execution tests'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='research_test_report.txt',
        help='Output file for report'
    )
    
    parser.add_argument(
        '--json-output',
        default='research_test_results.json',
        help='Output file for JSON results'
    )
    
    args = parser.parse_args()
    
    # Run tests
    runner = ResearchTestRunner(verbose=args.verbose)
    
    print("Starting HMM Research Environment Test Suite...")
    print(f"Include slow tests: {args.include_slow}")
    print(f"Include notebook tests: {args.include_notebooks}")
    
    results = runner.run_all_tests(
        include_slow=args.include_slow,
        include_notebooks=args.include_notebooks
    )
    
    # Generate and display report
    report = runner.generate_report()
    print("\n" + report)
    
    # Save reports
    runner.save_report(args.output)
    runner.save_json_results(args.json_output)
    
    # Exit with appropriate code
    all_passed = all(r['success'] for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
