#!/usr/bin/env python3
"""
Example script demonstrating backtesting data loading capabilities.

This script shows how to:
1. Load signal and market data from Parquet files
2. Validate data quality
3. Load HMM artifacts from MinIO with fallback
4. Use caching for performance
"""

import sys
from pathlib import Path
from datetime import date
import logging

# Add the imp package to the path
sys.path.append(str(Path(__file__).parent.parent))

from imp.backtesting import (
    DataLoader,
    ArtifactLoader,
    DataSourceConfig,
    DataValidationError,
    ArtifactLoadError
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    
    print("🚀 Backtesting Data Loading Example")
    print("=" * 50)
    
    # Example configuration
    config = DataSourceConfig(
        signals_path=Path("../rust/partitioned_data"),  # Adjust path as needed
        market_data_path=Path("../rust/partitioned_data"),  # Same for this example
        minio_endpoint="localhost:9000",
        minio_access_key="admin",
        minio_secret_key="admin123",
        minio_bucket="artifacts",
        local_cache_path=Path("./cache")
    )
    
    # Example 1: Data Loading
    print("\n📊 Example 1: Loading Signal and Market Data")
    print("-" * 40)
    
    try:
        # Initialize data loader
        data_loader = DataLoader(config)
        
        # Define test parameters
        symbols = ["BTCUSDT"]
        start_date = date(2025, 9, 19)
        end_date = date(2025, 9, 19)
        
        print(f"Loading data for symbols: {symbols}")
        print(f"Date range: {start_date} to {end_date}")
        
        # Load signal data
        try:
            signal_df = data_loader.load_signal_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                validate=True
            )
            print(f"✅ Signal data loaded: {signal_df.shape}")
            print(f"   Columns: {list(signal_df.columns)}")
            print(f"   Date range: {signal_df.index.min()} to {signal_df.index.max()}")
            
        except FileNotFoundError as e:
            print(f"⚠️  Signal data not found: {e}")
            signal_df = None
        except DataValidationError as e:
            print(f"❌ Signal data validation failed: {e}")
            signal_df = None
        
        # Load market data
        try:
            market_df = data_loader.load_market_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                validate=True
            )
            print(f"✅ Market data loaded: {market_df.shape}")
            print(f"   Columns: {list(market_df.columns)}")
            print(f"   Date range: {market_df.index.min()} to {market_df.index.max()}")
            
        except FileNotFoundError as e:
            print(f"⚠️  Market data not found: {e}")
            market_df = None
        except DataValidationError as e:
            print(f"❌ Market data validation failed: {e}")
            market_df = None
        
        # Load combined data if both are available
        if signal_df is not None and market_df is not None:
            try:
                combined_signal, combined_market = data_loader.load_combined_data(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    validate=True
                )
                print(f"✅ Combined data loaded:")
                print(f"   Signal records: {len(combined_signal)}")
                print(f"   Market records: {len(combined_market)}")
                
            except Exception as e:
                print(f"❌ Combined data loading failed: {e}")
        
    except Exception as e:
        print(f"❌ Data loading example failed: {e}")
    
    # Example 2: Data Quality Validation
    print("\n🔍 Example 2: Data Quality Validation")
    print("-" * 40)
    
    try:
        # Perform data quality validation
        quality_report = data_loader.validate_data_quality()
        
        print(f"📋 Data Quality Report:")
        print(f"   Quality Score: {quality_report.quality_score:.2f}")
        print(f"   Total Records: {quality_report.total_records:,}")
        print(f"   Symbols: {len(quality_report.symbols)}")
        print(f"   Date Range: {quality_report.date_range[0]} to {quality_report.date_range[1]}")
        
        if quality_report.warnings:
            print(f"   ⚠️  Warnings ({len(quality_report.warnings)}):")
            for warning in quality_report.warnings[:3]:  # Show first 3
                print(f"      - {warning}")
        
        if quality_report.errors:
            print(f"   ❌ Errors ({len(quality_report.errors)}):")
            for error in quality_report.errors[:3]:  # Show first 3
                print(f"      - {error}")
        
        # Save quality report
        report_path = Path("./data_quality_report.json")
        data_loader.save_quality_report(report_path)
        print(f"   💾 Report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Data quality validation failed: {e}")
    
    # Example 3: Artifact Loading
    print("\n🏺 Example 3: HMM Artifact Loading")
    print("-" * 40)
    
    try:
        # Initialize artifact loader
        artifact_loader = ArtifactLoader(config)
        
        # Check health
        health = artifact_loader.health_check()
        print(f"🏥 Health Check: {health['status']}")
        
        for check_name, check_result in health['checks'].items():
            status_emoji = "✅" if check_result['status'] == 'healthy' else "⚠️" if check_result['status'] == 'degraded' else "❌"
            print(f"   {status_emoji} {check_name}: {check_result['message']}")
        
        # List available artifacts
        print(f"\n📋 Available Artifacts:")
        artifacts = artifact_loader.list_available_artifacts()
        
        print(f"   MinIO: {len(artifacts['minio'])} artifacts")
        for artifact in artifacts['minio'][:3]:  # Show first 3
            print(f"      - {artifact['experiment_id']} v{artifact['version']} (tags: {artifact['tags']})")
        
        print(f"   Local: {len(artifacts['local'])} artifacts")
        for artifact in artifacts['local'][:3]:  # Show first 3
            print(f"      - {artifact['experiment_id']} v{artifact['version']} (tags: {artifact['tags']})")
        
        if artifacts['errors']:
            print(f"   ⚠️  Errors: {len(artifacts['errors'])}")
        
        # Try to load production artifact
        try:
            print(f"\n🎯 Loading Production Artifact:")
            weights_data = artifact_loader.load_hmm_weights(
                experiment_id=None,  # Production artifact
                version="latest",
                use_cache=True
            )
            
            metadata = weights_data['metadata']
            print(f"   ✅ Loaded from: {weights_data['source']}")
            print(f"   Experiment: {metadata.get('experiment_id', 'N/A')}")
            print(f"   Version: {metadata.get('version', 'N/A')}")
            print(f"   States: {metadata.get('n_states', 'N/A')}")
            print(f"   Library: {metadata.get('library_used', 'N/A')}")
            print(f"   Has fusion weights: {weights_data['fusion_weights'] is not None}")
            
        except ArtifactLoadError as e:
            print(f"   ⚠️  Production artifact not available: {e}")
        
        # Show cache statistics
        cache_stats = artifact_loader.get_cache_stats()
        print(f"\n💾 Cache Statistics:")
        print(f"   Size: {cache_stats['size']}/{cache_stats['max_size']}")
        print(f"   Expired entries: {cache_stats['expired_entries']}")
        
    except Exception as e:
        print(f"❌ Artifact loading example failed: {e}")
    
    # Example 4: Performance Testing
    print("\n⚡ Example 4: Performance Testing")
    print("-" * 40)
    
    try:
        import time
        
        # Test cache performance
        print("Testing cache performance...")
        
        # First load (cache miss)
        start_time = time.time()
        try:
            weights_data = artifact_loader.load_hmm_weights(
                experiment_id=None,
                version="latest",
                use_cache=True
            )
            first_load_time = time.time() - start_time
            print(f"   First load (cache miss): {first_load_time:.3f}s")
            
            # Second load (cache hit)
            start_time = time.time()
            weights_data = artifact_loader.load_hmm_weights(
                experiment_id=None,
                version="latest",
                use_cache=True
            )
            second_load_time = time.time() - start_time
            print(f"   Second load (cache hit): {second_load_time:.3f}s")
            
            if first_load_time > 0:
                speedup = first_load_time / second_load_time if second_load_time > 0 else float('inf')
                print(f"   Cache speedup: {speedup:.1f}x")
            
        except ArtifactLoadError:
            print("   ⚠️  No artifacts available for performance testing")
        
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
    
    print(f"\n✨ Example completed!")
    print(f"💡 Tips:")
    print(f"   - Ensure MinIO is running on localhost:9000 for full functionality")
    print(f"   - Place sample data in ../rust/partitioned_data/ directory")
    print(f"   - Check logs for detailed information about data loading")


if __name__ == "__main__":
    main()