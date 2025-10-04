"""
Unit tests for LDC data integration and preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from imp.data import (
    LDCDataLoader, LDCSignalConfig,
    SignalPreprocessor, PreprocessingConfig,
    DataValidator, ValidationReport,
    FeatureEngineer, FeatureConfig,
    DataQualityReporter
)


@pytest.fixture
def sample_data():
    """Create sample time series data for testing."""
    dates = pd.date_range('2024-01-01', periods=200, freq='5min')
    df = pd.DataFrame({
        's_LDC': np.random.randn(200).cumsum(),
        's_MR': np.random.randn(200) * 0.5,
        's_TSMOM': np.random.randn(200) * 0.8
    }, index=dates)
    df.index.name = 'timestamp'
    return df


@pytest.fixture
def sample_data_with_issues():
    """Create sample data with quality issues."""
    dates = pd.date_range('2024-01-01', periods=200, freq='5min')
    df = pd.DataFrame({
        'signal1': np.random.randn(200),
        'signal2': np.random.randn(200),
        'signal3': np.random.randn(200)
    }, index=dates)
    
    # Add missing values
    df.loc[df.index[10:15], 'signal1'] = np.nan
    
    # Add outliers
    df.loc[df.index[50], 'signal2'] = 100
    
    df.index.name = 'timestamp'
    return df


class TestLDCDataLoader:
    """Test LDCDataLoader functionality."""
    
    def test_loader_initialization(self):
        """Test loader initialization with config."""
        config = LDCSignalConfig(signals=['s_LDC', 's_MR'])
        loader = LDCDataLoader(config)
        
        assert loader.config.signals == ['s_LDC', 's_MR']
        assert loader._loaded_data is None
    
    def test_load_from_file(self, sample_data, tmp_path):
        """Test loading from parquet file."""
        # Save sample data
        file_path = tmp_path / 'test_data.parquet'
        sample_data.to_parquet(file_path)
        
        # Load data
        loader = LDCDataLoader()
        df = loader.load_from_file(file_path)
        
        assert len(df) == 200
        assert 's_LDC' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
    
    def test_get_signal_statistics(self, sample_data, tmp_path):
        """Test signal statistics computation."""
        file_path = tmp_path / 'test_data.parquet'
        sample_data.to_parquet(file_path)
        
        loader = LDCDataLoader()
        df = loader.load_from_file(file_path)
        stats = loader.get_signal_statistics()
        
        assert 'mean' in stats.columns
        assert 'std' in stats.columns
        assert len(stats) == len(df.columns)


class TestSignalPreprocessor:
    """Test SignalPreprocessor functionality."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        config = PreprocessingConfig(
            scaling_method='standardize',
            handle_missing='forward_fill'
        )
        preprocessor = SignalPreprocessor(config)
        
        assert preprocessor.config.scaling_method == 'standardize'
        assert preprocessor.config.handle_missing == 'forward_fill'
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        preprocessor = SignalPreprocessor()
        df_processed = preprocessor.fit_transform(sample_data)
        
        assert len(df_processed) == len(sample_data)
        assert df_processed.columns.tolist() == sample_data.columns.tolist()
        
        # Check standardization
        for col in df_processed.columns:
            assert abs(df_processed[col].mean()) < 0.1  # Close to 0
            assert abs(df_processed[col].std() - 1.0) < 0.1  # Close to 1
    
    def test_handle_missing_values(self, sample_data_with_issues):
        """Test missing value handling."""
        config = PreprocessingConfig(handle_missing='forward_fill')
        preprocessor = SignalPreprocessor(config)
        
        df_processed = preprocessor.fit_transform(sample_data_with_issues)
        
        # Should have no missing values after preprocessing
        assert df_processed.isna().sum().sum() == 0
    
    def test_outlier_detection(self, sample_data_with_issues):
        """Test outlier detection and handling."""
        config = PreprocessingConfig(
            outlier_method='zscore',
            outlier_threshold=3.0,
            outlier_action='clip'
        )
        preprocessor = SignalPreprocessor(config)
        
        df_processed = preprocessor.fit_transform(sample_data_with_issues)
        
        # Outliers should be clipped
        assert df_processed['signal2'].max() < 10  # Much less than original 100
    
    def test_get_recommendations(self, sample_data):
        """Test preprocessing recommendations."""
        preprocessor = SignalPreprocessor()
        recommendations = preprocessor.get_recommendations(sample_data)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestDataValidator:
    """Test DataValidator functionality."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = DataValidator(min_samples=100, max_missing_pct=5.0)
        
        assert validator.min_samples == 100
        assert validator.max_missing_pct == 5.0
    
    def test_validate_good_data(self, sample_data):
        """Test validation of good quality data."""
        validator = DataValidator()
        report = validator.validate(sample_data)
        
        assert isinstance(report, ValidationReport)
        assert report.is_valid
        assert len(report.checks_passed) > 0
    
    def test_validate_data_with_issues(self, sample_data_with_issues):
        """Test validation of data with issues."""
        validator = DataValidator(max_missing_pct=1.0)  # Strict threshold
        report = validator.validate(sample_data_with_issues)
        
        # Should detect missing data issue
        assert len(report.checks_failed) > 0 or len(report.warnings) > 0
    
    def test_quick_check(self, sample_data):
        """Test quick validation check."""
        validator = DataValidator()
        is_valid = validator.quick_check(sample_data)
        
        assert isinstance(is_valid, bool)
        assert is_valid  # Good data should pass


class TestFeatureEngineer:
    """Test FeatureEngineer functionality."""
    
    def test_engineer_initialization(self):
        """Test feature engineer initialization."""
        config = FeatureConfig(add_returns=True, add_rolling_stats=True)
        engineer = FeatureEngineer(config)
        
        assert engineer.config.add_returns
        assert engineer.config.add_rolling_stats
    
    def test_fit_transform(self, sample_data):
        """Test feature engineering."""
        config = FeatureConfig(
            add_returns=True,
            add_rolling_stats=True,
            rolling_windows=[5, 10]
        )
        engineer = FeatureEngineer(config)
        
        df_features = engineer.fit_transform(sample_data)
        
        # Should have more features than original
        assert len(df_features.columns) > len(sample_data.columns)
        
        # Check for expected features
        assert any('_return' in col for col in df_features.columns)
        assert any('_ma5' in col for col in df_features.columns)
    
    def test_get_feature_importance(self, sample_data):
        """Test feature importance calculation."""
        engineer = FeatureEngineer()
        df_features = engineer.fit_transform(sample_data)
        
        importance = engineer.get_feature_importance(df_features)
        
        assert 'feature' in importance.columns
        assert 'variance' in importance.columns
        assert len(importance) == len(df_features.columns)
    
    def test_select_top_features(self, sample_data):
        """Test feature selection."""
        config = FeatureConfig(add_returns=True, add_rolling_stats=True)
        engineer = FeatureEngineer(config)
        
        df_features = engineer.fit_transform(sample_data)
        df_top = engineer.select_top_features(df_features, n_features=5)
        
        assert len(df_top.columns) == 5


class TestDataQualityReporter:
    """Test DataQualityReporter functionality."""
    
    def test_reporter_initialization(self):
        """Test reporter initialization."""
        reporter = DataQualityReporter()
        assert reporter.report_data == {}
    
    def test_generate_report(self, sample_data):
        """Test report generation."""
        reporter = DataQualityReporter()
        report = reporter.generate_report(sample_data)
        
        assert 'timestamp' in report
        assert 'overview' in report
        assert 'quality_metrics' in report
        assert 'recommendations' in report
    
    def test_save_report_json(self, sample_data, tmp_path):
        """Test saving report as JSON."""
        reporter = DataQualityReporter()
        reporter.generate_report(sample_data)
        
        output_path = tmp_path / 'report.json'
        reporter.save_report(output_path, format='json')
        
        assert output_path.exists()
    
    def test_save_report_txt(self, sample_data, tmp_path):
        """Test saving report as text."""
        reporter = DataQualityReporter()
        reporter.generate_report(sample_data)
        
        output_path = tmp_path / 'report.txt'
        reporter.save_report(output_path, format='txt')
        
        assert output_path.exists()


class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_complete_pipeline(self, sample_data):
        """Test complete data processing pipeline."""
        # 1. Validate
        validator = DataValidator()
        validation_report = validator.validate(sample_data)
        assert validation_report.is_valid
        
        # 2. Preprocess
        preprocessor = SignalPreprocessor()
        df_processed = preprocessor.fit_transform(sample_data)
        assert len(df_processed) == len(sample_data)
        
        # 3. Engineer features
        engineer = FeatureEngineer(FeatureConfig(add_returns=True))
        df_features = engineer.fit_transform(df_processed)
        assert len(df_features.columns) > len(df_processed.columns)
        
        # 4. Generate report
        reporter = DataQualityReporter()
        report = reporter.generate_report(df_features)
        assert 'overview' in report
        
        # 5. Prepare for HMM
        observations = df_features.values
        assert observations.shape[0] > 0
        assert observations.shape[1] > 0
        assert not np.isnan(observations).any()
    
    def test_pipeline_with_problematic_data(self, sample_data_with_issues):
        """Test pipeline handles problematic data."""
        # Preprocess to fix issues
        preprocessor = SignalPreprocessor(
            PreprocessingConfig(
                handle_missing='forward_fill',
                outlier_action='clip'
            )
        )
        df_processed = preprocessor.fit_transform(sample_data_with_issues)
        
        # Should have no missing values or extreme outliers
        assert df_processed.isna().sum().sum() == 0
        assert df_processed.abs().max().max() < 10
        
        # Should be ready for HMM training
        observations = df_processed.values
        assert not np.isnan(observations).any()
        assert not np.isinf(observations).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
