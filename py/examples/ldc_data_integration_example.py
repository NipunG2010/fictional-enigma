"""
Example: LDC Signal Data Integration and Preprocessing Pipeline

This example demonstrates the complete workflow for loading, preprocessing,
validating, and engineering features from LDC signal data for HMM training.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from imp.data import (
    LDCDataLoader, LDCSignalConfig,
    SignalPreprocessor, PreprocessingConfig,
    DataValidator,
    FeatureEngineer, FeatureConfig,
    DataQualityReporter
)
import numpy as np
import pandas as pd


def example_basic_loading():
    """Example 1: Basic data loading from Rust LDC output."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Data Loading")
    print("="*70)
    
    # Load data with default configuration
    loader = LDCDataLoader()
    
    # Try loading from sample directory
    sample_dir = project_root / 'rust' / 'sample'
    if sample_dir.exists():
        df = loader.load_from_directory(sample_dir)
        print(f"\n✓ Loaded {len(df)} samples with {len(df.columns)} features")
        print(f"✓ Features: {df.columns.tolist()}")
        
        # Get statistics
        stats = loader.get_signal_statistics()
        print("\n📊 Signal Statistics:")
        print(stats[['mean', 'std', 'min', 'max', 'missing']].to_string())
        
        return df
    else:
        print(f"⚠️  Sample directory not found: {sample_dir}")
        print("   Generating synthetic data instead...")
        
        # Generate synthetic data
        dates = pd.date_range('2024-01-01', periods=500, freq='5T')
        df = pd.DataFrame({
            's_LDC': np.random.randn(500).cumsum(),
            's_MR': np.random.randn(500) * 0.5,
            's_TSMOM': np.random.randn(500) * 0.8
        }, index=dates)
        df.index.name = 'timestamp'
        
        print(f"✓ Generated {len(df)} synthetic samples")
        return df


def example_advanced_loading():
    """Example 2: Advanced loading with configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Advanced Loading with Configuration")
    print("="*70)
    
    # Configure specific signals and date range
    config = LDCSignalConfig(
        signals=['s_LDC', 's_MR', 's_TSMOM'],
        features=['rsi', 'cci', 'adx'],
        start_date=pd.Timestamp('2024-01-01'),
        end_date=pd.Timestamp('2024-12-31')
    )
    
    loader = LDCDataLoader(config)
    
    # Try loading
    sample_dir = project_root / 'rust' / 'sample'
    if sample_dir.exists():
        df = loader.load_from_directory(sample_dir)
    else:
        # Generate synthetic data with requested features
        dates = pd.date_range('2024-01-01', periods=500, freq='5T')
        df = pd.DataFrame({
            's_LDC': np.random.randn(500).cumsum(),
            's_MR': np.random.randn(500) * 0.5,
            's_TSMOM': np.random.randn(500) * 0.8,
            'rsi': np.random.uniform(20, 80, 500),
            'cci': np.random.randn(500) * 100,
            'adx': np.random.uniform(10, 50, 500)
        }, index=dates)
        df.index.name = 'timestamp'
    
    print(f"✓ Loaded data with configuration")
    print(f"  Signals: {[c for c in df.columns if c.startswith('s_')]}")
    print(f"  Features: {[c for c in df.columns if not c.startswith('s_')]}")
    
    return df


def example_data_validation(df):
    """Example 3: Data validation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Data Validation")
    print("="*70)
    
    # Create validator with custom settings
    validator = DataValidator(
        min_samples=100,
        max_missing_pct=10.0,
        check_stationarity=True,
        check_multicollinearity=True
    )
    
    # Validate data
    report = validator.validate(df)
    
    # Print summary
    report.print_summary()
    
    # Quick check
    is_valid = validator.quick_check(df)
    print(f"\n✓ Quick validation: {'PASSED' if is_valid else 'FAILED'}")
    
    return report


def example_preprocessing(df):
    """Example 4: Signal preprocessing."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Signal Preprocessing")
    print("="*70)
    
    # Configure preprocessing
    config = PreprocessingConfig(
        scaling_method='standardize',
        handle_missing='forward_fill',
        outlier_method='zscore',
        outlier_threshold=3.0,
        outlier_action='clip',
        remove_trend=False
    )
    
    # Create preprocessor
    preprocessor = SignalPreprocessor(config)
    
    # Get recommendations first
    print("\n💡 Preprocessing Recommendations:")
    recommendations = preprocessor.get_recommendations(df)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Preprocess data
    print("\n🔧 Applying preprocessing...")
    df_processed = preprocessor.fit_transform(df)
    
    # Get statistics
    stats = preprocessor.get_preprocessing_stats()
    print(f"\n✓ Preprocessing complete:")
    print(f"  Input shape: {stats['input_shape']}")
    print(f"  Output shape: {stats['output_shape']}")
    print(f"  Steps applied: {len(stats['steps'])}")
    
    for step in stats['steps']:
        print(f"    • {step['step']}: {step.get('method', 'N/A')}")
    
    return df_processed, preprocessor


def example_feature_engineering(df):
    """Example 5: Feature engineering."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Feature Engineering")
    print("="*70)
    
    # Configure feature engineering
    config = FeatureConfig(
        add_returns=True,
        add_rolling_stats=True,
        rolling_windows=[5, 10, 20],
        add_momentum=True,
        add_volatility=True,
        add_lags=False,
        add_pca=False,
        smooth_signals=False
    )
    
    # Create feature engineer
    engineer = FeatureEngineer(config)
    
    # Engineer features
    df_features = engineer.fit_transform(df)
    
    print(f"\n✓ Feature engineering complete:")
    print(f"  Original features: {len(df.columns)}")
    print(f"  Engineered features: {len(df_features.columns)}")
    print(f"  Total samples: {len(df_features)}")
    
    # Get feature importance
    print("\n📊 Top 10 Features by Variance:")
    importance = engineer.get_feature_importance(df_features)
    print(importance.head(10)[['feature', 'variance', 'mean_abs']].to_string())
    
    return df_features, engineer


def example_quality_reporting(df, validation_report, preprocessing_stats, feature_importance):
    """Example 6: Data quality reporting."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Data Quality Reporting")
    print("="*70)
    
    # Create reporter
    reporter = DataQualityReporter()
    
    # Generate comprehensive report
    report = reporter.generate_report(
        df=df,
        validation_report=validation_report,
        preprocessing_stats=preprocessing_stats,
        feature_importance=feature_importance
    )
    
    # Print report
    reporter.print_report(detailed=True)
    
    # Save reports
    output_dir = project_root / 'processed_data'
    output_dir.mkdir(exist_ok=True)
    
    reporter.save_report(output_dir / 'quality_report.json', format='json')
    print(f"\n✓ Report saved to {output_dir / 'quality_report.json'}")
    
    # Create visualization dashboard
    try:
        fig = reporter.plot_quality_dashboard(df, save_path=output_dir / 'quality_dashboard.png')
        print(f"✓ Dashboard saved to {output_dir / 'quality_dashboard.png'}")
    except Exception as e:
        print(f"⚠️  Could not create dashboard: {e}")
    
    return report


def example_complete_pipeline():
    """Example 7: Complete end-to-end pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Complete End-to-End Pipeline")
    print("="*70)
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    loader_config = LDCSignalConfig(
        signals=['s_LDC', 's_MR', 's_TSMOM']
    )
    loader = LDCDataLoader(loader_config)
    
    sample_dir = project_root / 'rust' / 'sample'
    if sample_dir.exists():
        df = loader.load_from_directory(sample_dir)
    else:
        # Generate synthetic data
        dates = pd.date_range('2024-01-01', periods=500, freq='5T')
        df = pd.DataFrame({
            's_LDC': np.random.randn(500).cumsum(),
            's_MR': np.random.randn(500) * 0.5,
            's_TSMOM': np.random.randn(500) * 0.8
        }, index=dates)
        df.index.name = 'timestamp'
    
    print(f"✓ Loaded {len(df)} samples")
    
    # Step 2: Validate
    print("\n[2/6] Validating data...")
    validator = DataValidator()
    validation_report = validator.validate(df)
    print(f"✓ Validation: {'PASSED' if validation_report.is_valid else 'FAILED'}")
    
    # Step 3: Preprocess
    print("\n[3/6] Preprocessing...")
    preproc_config = PreprocessingConfig(
        scaling_method='standardize',
        handle_missing='forward_fill',
        outlier_method='zscore',
        outlier_action='clip'
    )
    preprocessor = SignalPreprocessor(preproc_config)
    df_processed = preprocessor.fit_transform(df)
    print(f"✓ Preprocessed: {df_processed.shape}")
    
    # Step 4: Engineer features
    print("\n[4/6] Engineering features...")
    feature_config = FeatureConfig(
        add_returns=True,
        add_rolling_stats=True,
        rolling_windows=[5, 10],
        add_momentum=True
    )
    engineer = FeatureEngineer(feature_config)
    df_features = engineer.fit_transform(df_processed)
    print(f"✓ Engineered: {df_features.shape}")
    
    # Step 5: Generate report
    print("\n[5/6] Generating quality report...")
    reporter = DataQualityReporter()
    report = reporter.generate_report(
        df=df_features,
        validation_report=validation_report,
        preprocessing_stats=preprocessor.get_preprocessing_stats(),
        feature_importance=engineer.get_feature_importance(df_features)
    )
    print(f"✓ Report generated")
    
    # Step 6: Prepare for HMM
    print("\n[6/6] Preparing for HMM training...")
    observations = df_features.values
    print(f"✓ Observations shape: {observations.shape}")
    print(f"✓ Data type: {observations.dtype}")
    print(f"✓ Value range: [{observations.min():.3f}, {observations.max():.3f}]")
    
    # Save processed data
    output_dir = project_root / 'processed_data'
    output_dir.mkdir(exist_ok=True)
    
    df_features.to_parquet(output_dir / 'hmm_observations.parquet')
    np.save(output_dir / 'observations.npy', observations)
    
    print(f"\n✓ Saved processed data to {output_dir}")
    print("\n🎉 Pipeline complete! Ready for HMM training.")
    
    return df_features, observations


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("LDC SIGNAL DATA INTEGRATION AND PREPROCESSING EXAMPLES")
    print("="*70)
    
    # Example 1: Basic loading
    df = example_basic_loading()
    
    # Example 2: Advanced loading
    df = example_advanced_loading()
    
    # Example 3: Validation
    validation_report = example_data_validation(df)
    
    # Example 4: Preprocessing
    df_processed, preprocessor = example_preprocessing(df)
    
    # Example 5: Feature engineering
    df_features, engineer = example_feature_engineering(df_processed)
    
    # Example 6: Quality reporting
    example_quality_reporting(
        df_features,
        validation_report,
        preprocessor.get_preprocessing_stats(),
        engineer.get_feature_importance(df_features)
    )
    
    # Example 7: Complete pipeline
    df_final, observations = example_complete_pipeline()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the quality report in processed_data/quality_report.json")
    print("2. Check the visualization dashboard in processed_data/quality_dashboard.png")
    print("3. Use the observations array for HMM training:")
    print("   from imp.hmm import EnhancedHMMTrainer")
    print("   trainer = EnhancedHMMTrainer(n_states=3)")
    print("   artifact, metrics = trainer.train_with_validation(observations)")
    print()


if __name__ == '__main__':
    main()
