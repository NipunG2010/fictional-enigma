"""Environment validation utilities for HMM research environment.

This module provides comprehensive validation of the development environment,
checking for required dependencies, versions, and configurations.
"""

import sys
import importlib.util
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings


@dataclass
class PackageInfo:
    """Information about a package requirement."""
    
    name: str
    import_name: str
    min_version: Optional[str] = None
    required: bool = True
    purpose: str = ""


@dataclass
class ValidationResult:
    """Result of environment validation."""
    
    package: str
    installed: bool
    version: Optional[str] = None
    meets_requirements: bool = True
    message: str = ""


class EnvironmentValidator:
    """Validates the research environment setup."""
    
    # Core dependencies
    CORE_PACKAGES = [
        PackageInfo("numpy", "numpy", "1.24.0", True, "Numerical computing"),
        PackageInfo("pandas", "pandas", "2.0.0", True, "Data manipulation"),
        PackageInfo("polars", "polars", "0.20.0", True, "Fast data processing"),
        PackageInfo("scikit-learn", "sklearn", "1.3.0", True, "Machine learning"),
        PackageInfo("pydantic", "pydantic", "2.0.0", True, "Data validation"),
    ]
    
    # HMM-specific packages
    HMM_PACKAGES = [
        PackageInfo("hmmlearn", "hmmlearn", "0.3.0", True, "HMM implementation (hmmlearn)"),
        PackageInfo("pomegranate", "pomegranate", "0.14.0", True, "HMM implementation (pomegranate)"),
    ]
    
    # Jupyter and notebook packages
    JUPYTER_PACKAGES = [
        PackageInfo("jupyter", "jupyter", "1.0.0", True, "Jupyter notebook"),
        PackageInfo("jupyterlab", "jupyterlab", "4.0.0", False, "JupyterLab interface"),
        PackageInfo("ipykernel", "ipykernel", "6.25.0", True, "Jupyter kernel"),
        PackageInfo("ipywidgets", "ipywidgets", "8.0.0", True, "Interactive widgets"),
        PackageInfo("nbformat", "nbformat", "5.9.0", True, "Notebook format"),
    ]
    
    # Visualization packages
    VIZ_PACKAGES = [
        PackageInfo("matplotlib", "matplotlib", "3.7.0", True, "Static plotting"),
        PackageInfo("seaborn", "seaborn", "0.12.0", True, "Statistical visualization"),
        PackageInfo("plotly", "plotly", "5.15.0", True, "Interactive plotting"),
    ]
    
    # Testing packages
    TEST_PACKAGES = [
        PackageInfo("pytest", "pytest", "7.4.0", True, "Testing framework"),
        PackageInfo("pytest-asyncio", "pytest_asyncio", "0.21.0", False, "Async testing"),
        PackageInfo("pytest-cov", "pytest_cov", "4.1.0", False, "Coverage reporting"),
    ]
    
    # Optional optimization packages
    OPT_PACKAGES = [
        PackageInfo("scikit-optimize", "skopt", "0.9.0", False, "Bayesian optimization"),
        PackageInfo("joblib", "joblib", "1.3.0", False, "Parallel processing"),
    ]
    
    def __init__(self):
        """Initialize the environment validator."""
        self.results: List[ValidationResult] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Check if Python version meets requirements."""
        version = sys.version_info
        required = (3, 9)
        
        if version >= required:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"Python {version.major}.{version.minor}.{version.micro} (requires >= 3.9)"
    
    def check_package(self, package_info: PackageInfo) -> ValidationResult:
        """Check if a package is installed and meets version requirements."""
        spec = importlib.util.find_spec(package_info.import_name)
        
        if spec is None:
            return ValidationResult(
                package=package_info.name,
                installed=False,
                meets_requirements=False,
                message=f"Not installed - {package_info.purpose}"
            )
        
        # Try to get version
        try:
            module = importlib.import_module(package_info.import_name)
            version = getattr(module, "__version__", "unknown")
            
            # Check version if specified
            meets_req = True
            if package_info.min_version and version != "unknown":
                try:
                    from packaging import version as pkg_version
                    meets_req = pkg_version.parse(version) >= pkg_version.parse(package_info.min_version)
                except ImportError:
                    # If packaging is not available, skip version check
                    meets_req = True
            
            return ValidationResult(
                package=package_info.name,
                installed=True,
                version=version,
                meets_requirements=meets_req,
                message=f"Installed (v{version})" if meets_req else f"Version {version} < {package_info.min_version}"
            )
        except Exception as e:
            return ValidationResult(
                package=package_info.name,
                installed=True,
                version="unknown",
                meets_requirements=True,
                message=f"Installed (version check failed: {str(e)})"
            )
    
    def validate_package_group(self, packages: List[PackageInfo], group_name: str) -> None:
        """Validate a group of packages."""
        print(f"\n{group_name}:")
        print("-" * 60)
        
        for package_info in packages:
            result = self.check_package(package_info)
            self.results.append(result)
            
            # Format output
            status = "✓" if result.installed and result.meets_requirements else "✗"
            required_marker = "[REQUIRED]" if package_info.required else "[OPTIONAL]"
            
            print(f"  {status} {package_info.name:25} {required_marker:12} {result.message}")
            
            # Track errors and warnings
            if package_info.required and not result.installed:
                self.errors.append(f"Required package '{package_info.name}' is not installed")
            elif package_info.required and not result.meets_requirements:
                self.warnings.append(f"Package '{package_info.name}' version may be incompatible")
    
    def check_jupyter_kernel(self) -> bool:
        """Check if Jupyter kernel is properly configured."""
        try:
            import jupyter_client
            km = jupyter_client.kernelspec.KernelSpecManager()
            kernels = km.get_all_specs()
            
            if "imp-research" in kernels:
                print("  ✓ Jupyter kernel 'imp-research' is configured")
                return True
            else:
                self.warnings.append("Jupyter kernel 'imp-research' is not configured")
                print("  ✗ Jupyter kernel 'imp-research' is not configured")
                print("    Run: python -m ipykernel install --user --name=imp-research")
                return False
        except Exception as e:
            self.warnings.append(f"Could not check Jupyter kernel: {str(e)}")
            print(f"  ! Could not check Jupyter kernel: {str(e)}")
            return False
    
    def check_directory_structure(self) -> bool:
        """Check if required directories exist."""
        required_dirs = [
            "notebooks",
            "notebooks/utils",
            "processed_data",
            "tests",
        ]
        
        print("\nDirectory Structure:")
        print("-" * 60)
        
        all_exist = True
        for dir_path in required_dirs:
            path = Path(dir_path)
            exists = path.exists() and path.is_dir()
            status = "✓" if exists else "✗"
            print(f"  {status} {dir_path}")
            
            if not exists:
                all_exist = False
                self.warnings.append(f"Directory '{dir_path}' does not exist")
        
        return all_exist
    
    def validate_all(self) -> bool:
        """Run complete environment validation."""
        print("=" * 60)
        print("IMP Research Environment Validation")
        print("=" * 60)
        
        # Check Python version
        print("\nPython Version:")
        print("-" * 60)
        py_ok, py_msg = self.check_python_version()
        status = "✓" if py_ok else "✗"
        print(f"  {status} {py_msg}")
        
        if not py_ok:
            self.errors.append("Python version is too old")
        
        # Validate package groups
        self.validate_package_group(self.CORE_PACKAGES, "Core Dependencies")
        self.validate_package_group(self.HMM_PACKAGES, "HMM Libraries")
        self.validate_package_group(self.JUPYTER_PACKAGES, "Jupyter Environment")
        self.validate_package_group(self.VIZ_PACKAGES, "Visualization Libraries")
        self.validate_package_group(self.TEST_PACKAGES, "Testing Framework")
        self.validate_package_group(self.OPT_PACKAGES, "Optimization Tools")
        
        # Check Jupyter kernel
        print("\nJupyter Configuration:")
        print("-" * 60)
        self.check_jupyter_kernel()
        
        # Check directory structure
        self.check_directory_structure()
        
        # Summary
        print("\n" + "=" * 60)
        print("Validation Summary")
        print("=" * 60)
        
        total_packages = len(self.results)
        installed = sum(1 for r in self.results if r.installed)
        required_packages = len([p for group in [
            self.CORE_PACKAGES, self.HMM_PACKAGES, self.JUPYTER_PACKAGES,
            self.VIZ_PACKAGES, self.TEST_PACKAGES
        ] for p in group if p.required])
        required_installed = sum(1 for r in self.results if r.installed and r.meets_requirements)
        
        print(f"\nPackages: {installed}/{total_packages} installed")
        print(f"Required: {required_installed}/{required_packages} satisfied")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        success = len(self.errors) == 0
        
        if success:
            print("\n✅ Environment validation passed!")
            print("\nYou can now:")
            print("  1. Start Jupyter Lab: jupyter lab")
            print("  2. Run tests: pytest tests/")
            print("  3. Open example notebooks in notebooks/")
        else:
            print("\n❌ Environment validation failed!")
            print("\nPlease fix the errors above and run validation again.")
        
        print("=" * 60)
        
        return success
    
    def get_summary(self) -> Dict[str, any]:
        """Get validation summary as a dictionary."""
        return {
            "total_packages": len(self.results),
            "installed": sum(1 for r in self.results if r.installed),
            "errors": self.errors,
            "warnings": self.warnings,
            "success": len(self.errors) == 0
        }


def validate_environment() -> bool:
    """Convenience function to validate the environment."""
    validator = EnvironmentValidator()
    return validator.validate_all()


if __name__ == "__main__":
    # Run validation when executed as a script
    success = validate_environment()
    sys.exit(0 if success else 1)
