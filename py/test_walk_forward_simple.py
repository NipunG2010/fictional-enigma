"""Simple test to verify walk-forward validation implementation."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add py directory to path
sys.path.insert(0, str(Path(__file__).parent))

from imp.backtesting.walk_forward_validator import WalkForwardValidator, ValidationWindow
from imp.backtesting.config import WalkForwardConfig

def test_window_setup():
    """Test validation window setup."""
    print("Testing window setup...")
    
    config = WalkForwardConfig(
        enabled=True,
        train_period="6M",
        test_period="1M",
        step_size="2W",
        min_train_samples=1000,
        retrain_threshold=0.05
    )
    
    validator = WalkForwardValidator(config)
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    windows = validator.setup_windows(start_date, end_date)
    
    assert len(windows) > 0, "Should create at least one window"
    assert all(isinstance(w, ValidationWindow) for w in windows), "All windows should be ValidationWindow instances"
    
    # Check temporal ordering
    for window in windows:
        assert window.train_start < window.train_end, "Train start should be before train end"
        assert window.train_end == window.test_start, "Train end should equal test start"
        assert window.test_start < window.test_end, "Test start should be before test end"
    
    print(f"✓ Created {len(windows)} windows successfully")
    print(f"  First window: train=[{windows[0].train_start.date()}, {windows[0].train_end.date()}], "
          f"test=[{windows[0].test_start.date()}, {windows[0].test_end.date()}]")
    
    return True

def test_period_parsing():
    """Test period string parsing."""
    print("\nTesting period parsing...")
    
    config = WalkForwardConfig(
        enabled=True,
        train_period="1W",
        test_period="1D",
        step_size="1D"
    )
    
    validator = WalkForwardValidator(config)
    
    # Test various period formats
    assert validator._parse_period("1D") == timedelta(days=1)
    assert validator._parse_period("1W") == timedelta(weeks=1)
    assert validator._parse_period("1M") == timedelta(days=30)
    assert validator._parse_period("1Y") == timedelta(days=365)
    
    print("✓ Period parsing works correctly")
    
    return True

def test_temporal_separation():
    """Test temporal separation validation."""
    print("\nTesting temporal separation...")
    
    import pandas as pd
    
    config = WalkForwardConfig(enabled=True)
    validator = WalkForwardValidator(config)
    
    # Valid separation
    train_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', '2023-06-30', freq='D')
    })
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-07-01', '2023-07-31', freq='D')
    })
    
    is_valid = validator.validate_temporal_separation(train_data, test_data)
    assert is_valid, "Valid temporal separation should return True"
    
    # Invalid separation (overlap)
    test_data_overlap = pd.DataFrame({
        'timestamp': pd.date_range('2023-06-15', '2023-07-15', freq='D')
    })
    
    is_invalid = validator.validate_temporal_separation(train_data, test_data_overlap)
    assert not is_invalid, "Overlapping data should return False"
    
    print("✓ Temporal separation validation works correctly")
    
    return True

def test_window_serialization():
    """Test window serialization to dict."""
    print("\nTesting window serialization...")
    
    window = ValidationWindow(
        window_id=0,
        train_start=datetime(2023, 1, 1),
        train_end=datetime(2023, 6, 30),
        test_start=datetime(2023, 7, 1),
        test_end=datetime(2023, 7, 31),
        model_version="test_v1",
        retrained=True
    )
    
    window_dict = window.to_dict()
    
    assert window_dict['window_id'] == 0
    assert window_dict['model_version'] == "test_v1"
    assert window_dict['retrained'] == True
    assert 'train_start' in window_dict
    assert 'test_end' in window_dict
    
    print("✓ Window serialization works correctly")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Walk-Forward Validation Implementation Tests")
    print("=" * 60)
    
    tests = [
        test_window_setup,
        test_period_parsing,
        test_temporal_separation,
        test_window_serialization,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
