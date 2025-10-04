"""
Common utilities for Jupyter notebooks in the HMM research environment.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

# Add the project root to Python path for imports
def setup_notebook_environment():
    """Set up the notebook environment with proper imports and settings."""
    
    # Add project root to path
    project_root = Path.cwd()
    if 'notebooks' in str(Path.cwd()):
        project_root = Path.cwd().parent
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Configure warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print("✓ Notebook environment configured")
    print(f"✓ Project root: {project_root}")
    
    return project_root


def validate_data_quality(data: np.ndarray, 
                         name: str = "data") -> Dict[str, Any]:
    """Validate data quality and return diagnostic information."""
    
    diagnostics = {
        'name': name,
        'shape': data.shape,
        'dtype': str(data.dtype),
        'has_nan': np.isnan(data).any(),
        'has_inf': np.isinf(data).any(),
        'nan_count': np.isnan(data).sum(),
        'inf_count': np.isinf(data).sum(),
        'min_value': np.nanmin(data),
        'max_value': np.nanmax(data),
        'mean_value': np.nanmean(data),
        'std_value': np.nanstd(data)
    }
    
    return diagnostics


def print_data_diagnostics(diagnostics: Dict[str, Any]):
    """Print formatted data diagnostics."""
    
    print(f"\n📊 Data Quality Report: {diagnostics['name']}")
    print(f"   Shape: {diagnostics['shape']}")
    print(f"   Type: {diagnostics['dtype']}")
    print(f"   Range: [{diagnostics['min_value']:.4f}, {diagnostics['max_value']:.4f}]")
    print(f"   Mean ± Std: {diagnostics['mean_value']:.4f} ± {diagnostics['std_value']:.4f}")
    
    if diagnostics['has_nan']:
        print(f"   ⚠️  NaN values: {diagnostics['nan_count']}")
    else:
        print(f"   ✓ No NaN values")
        
    if diagnostics['has_inf']:
        print(f"   ⚠️  Infinite values: {diagnostics['inf_count']}")
    else:
        print(f"   ✓ No infinite values")


def create_progress_callback(description: str = "Processing"):
    """Create a progress callback for long-running operations."""
    
    def progress_callback(current: int, total: int):
        percent = (current / total) * 100
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f'\r{description}: |{bar}| {percent:.1f}% ({current}/{total})', end='')
        if current == total:
            print()  # New line when complete
    
    return progress_callback


def format_training_results(results: Dict[str, Any]) -> str:
    """Format training results for display."""
    
    formatted = f"""
    🎯 Training Results
    ==================
    Log-likelihood: {results.get('log_likelihood', 'N/A'):.4f}
    AIC: {results.get('aic', 'N/A'):.4f}
    BIC: {results.get('bic', 'N/A'):.4f}
    Converged: {results.get('converged', 'Unknown')}
    Iterations: {results.get('n_iter', 'N/A')}
    """
    
    return formatted


def safe_import(module_name: str, package_name: Optional[str] = None) -> tuple:
    """Safely import a module and return (module, success_flag)."""
    
    try:
        if package_name:
            module = __import__(module_name, fromlist=[package_name])
            return getattr(module, package_name), True
        else:
            module = __import__(module_name)
            return module, True
    except ImportError as e:
        print(f"⚠️  Failed to import {module_name}: {e}")
        return None, False


def check_dependencies():
    """Check if all required dependencies are available."""
    
    required_packages = [
        ('numpy', None),
        ('pandas', None),
        ('matplotlib', None),
        ('seaborn', None),
        ('sklearn', None),
        ('hmmlearn', None),
        ('ipywidgets', None)
    ]
    
    optional_packages = [
        ('plotly', None),
        ('pomegranate', None)
    ]
    
    print("🔍 Checking dependencies...")
    
    missing_required = []
    missing_optional = []
    
    for package, submodule in required_packages:
        _, success = safe_import(package, submodule)
        if not success:
            missing_required.append(package)
    
    for package, submodule in optional_packages:
        _, success = safe_import(package, submodule)
        if not success:
            missing_optional.append(package)
    
    if not missing_required:
        print("✓ All required dependencies available")
    else:
        print(f"❌ Missing required packages: {', '.join(missing_required)}")
    
    if missing_optional:
        print(f"⚠️  Missing optional packages: {', '.join(missing_optional)}")
    
    return len(missing_required) == 0