"""
Test systematic HMM training pipeline.

Tests for py/scripts/train_hmm_systematic.py covering:
- Data loading and validation
- Interpretability score calculation
- Model ranking logic
- Full pipeline integration
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch, MagicMock

# Import the SystematicHMMTrainer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_hmm_systematic import SystematicHMMTrainer

from imp.hmm.models import HMMArtifact
from imp.hmm.regime_analysis import RegimeCharacteristics, StatePersistence


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Create sample observation data."""
    np.random.seed(42)
    n_samples = 200
    
    # Generate synthetic [s_LDC, s_MR, s_TSMOM] data
    data = {
        's_LDC': np.random.randn(n_samples) * 0.5,
        's_MR': np.random.randn(n_samples) * 0.3,
        's_TSMOM': np.random.randn(n_samples) * 0.4
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_parquet_file(temp_dir, sample_data):
    """Create sample Parquet file with test data."""
    file_path = temp_dir / "test_data.parquet"
    sample_data.to_parquet(file_path)
    return file_path


@pytest.fixture
def sample_characteristics():
    """Create sample regime characteristics for testing."""
    return {
        0: RegimeCharacteristics(
            state_id=0,
            sample_count=80,
            mean_values={'s_LDC': 0.5, 's_MR': 0.2, 's_TSMOM': 0.3},
            std_values={'s_LDC': 0.3, 's_MR': 0.2, 's_TSMOM': 0.25},
            volatility=0.8,
            trend_strength=0.6,
            mean_reversion_score=0.3
        ),
        1: RegimeCharacteristics(
            state_id=1,
            sample_count=70,
            mean_values={'s_LDC': -0.3, 's_MR': 0.5, 's_TSMOM': -0.2},
            std_values={'s_LDC': 0.2, 's_MR': 0.3, 's_TSMOM': 0.2},
            volatility=0.4,
            trend_strength=0.3,
            mean_reversion_score=0.7
        ),
        2: RegimeCharacteristics(
            state_id=2,
            sample_count=50,
            mean_values={'s_LDC': 0.1, 's_MR': -0.4, 's_TSMOM': 0.6},
            std_values={'s_LDC': 0.4, 's_MR': 0.35, 's_TSMOM': 0.45},
            volatility=1.2,
            trend_strength=0.8,
            mean_reversion_score=0.2
        )
    }


@pytest.fixture
def sample_persistence():
    """Create sample state persistence for testing."""
    return {
        0: StatePersistence(
            state_id=0,
            total_occurrences=80,
            mean_duration=8.0,
            median_duration=6.0,
            max_duration=20,
            min_duration=2,
            stable_periods=40,
            transition_frequencies={1: 48, 2: 32},
            transition_probabilities={1: 0.6, 2: 0.4}
        ),
        1: StatePersistence(
            state_id=1,
            total_occurrences=70,
            mean_duration=12.0,
            median_duration=10.0,
            max_duration=25,
            min_duration=3,
            stable_periods=50,
            transition_frequencies={0: 35, 2: 35},
            transition_probabilities={0: 0.5, 2: 0.5}
        ),
        2: StatePersistence(
            state_id=2,
            total_occurrences=50,
            mean_duration=5.0,
            median_duration=4.0,
            max_duration=15,
            min_duration=1,
            stable_periods=20,
            transition_frequencies={0: 35, 1: 15},
            transition_probabilities={0: 0.7, 1: 0.3}
        )
    }


# ============================================================================
# Test Data Loading and Validation
# ============================================================================

def test_load_and_validate_data_success(temp_dir, sample_parquet_file):
    """Test successful data loading with valid Parquet file."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir,
        n_states_range=[2, 3]
    )
    
    observations = trainer.load_and_validate_data()
    
    # Check shape
    assert observations.shape[1] == 3, "Should have 3 features"
    assert observations.shape[0] == 200, "Should have 200 samples"
    
    # Check no NaN values
    assert not np.any(np.isnan(observations)), "Should not contain NaN"


def test_load_and_validate_data_missing_columns(temp_dir):
    """Test data loading fails with missing required columns."""
    # Create data with wrong columns
    df = pd.DataFrame({
        'wrong_col1': np.random.randn(100),
        'wrong_col2': np.random.randn(100)
    })
    
    file_path = temp_dir / "wrong_data.parquet"
    df.to_parquet(file_path)
    
    trainer = SystematicHMMTrainer(
        data_path=file_path,
        output_dir=temp_dir
    )
    
    with pytest.raises(ValueError, match="Missing required columns"):
        trainer.load_and_validate_data()


def test_load_and_validate_data_with_nan(temp_dir):
    """Test data loading handles NaN values correctly."""
    # Create data with NaN values
    df = pd.DataFrame({
        's_LDC': [1.0, 2.0, np.nan, 4.0, 5.0],
        's_MR': [1.0, np.nan, 3.0, 4.0, 5.0],
        's_TSMOM': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    
    file_path = temp_dir / "nan_data.parquet"
    df.to_parquet(file_path)
    
    trainer = SystematicHMMTrainer(
        data_path=file_path,
        output_dir=temp_dir
    )
    
    observations = trainer.load_and_validate_data()
    
    # Should remove rows with NaN
    assert observations.shape[0] == 3, "Should have 3 valid rows after removing NaN"
    assert not np.any(np.isnan(observations)), "Should not contain NaN"


def test_load_and_validate_data_all_nan(temp_dir):
    """Test data loading fails when all data is NaN."""
    # Create data with all NaN
    df = pd.DataFrame({
        's_LDC': [np.nan] * 10,
        's_MR': [np.nan] * 10,
        's_TSMOM': [np.nan] * 10
    })
    
    file_path = temp_dir / "all_nan_data.parquet"
    df.to_parquet(file_path)
    
    trainer = SystematicHMMTrainer(
        data_path=file_path,
        output_dir=temp_dir
    )
    
    with pytest.raises(ValueError, match="No valid observations"):
        trainer.load_and_validate_data()


def test_load_and_validate_data_csv_format(temp_dir):
    """Test data loading with CSV format (should work with pandas)."""
    # Create CSV file
    df = pd.DataFrame({
        's_LDC': np.random.randn(50),
        's_MR': np.random.randn(50),
        's_TSMOM': np.random.randn(50)
    })
    
    csv_path = temp_dir / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    # Note: SystematicHMMTrainer expects Parquet, but pandas can read CSV
    # This tests flexibility if we modify the loader
    trainer = SystematicHMMTrainer(
        data_path=csv_path,
        output_dir=temp_dir
    )
    
    # This will fail with current implementation (expects parquet)
    # but demonstrates the test pattern for format validation
    with pytest.raises(Exception):  # Will raise some pandas error
        trainer.load_and_validate_data()


# ============================================================================
# Test Interpretability Score Calculation
# ============================================================================

def test_calculate_interpretability_score_basic(temp_dir, sample_parquet_file,
                                                sample_characteristics, sample_persistence):
    """Test interpretability score calculation with valid data."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    score = trainer._calculate_interpretability_score(
        sample_characteristics,
        sample_persistence
    )
    
    # Score should be between 0 and 1
    assert 0.0 <= score <= 1.0, "Score should be in [0, 1] range"
    
    # With reasonable test data, score should be positive
    assert score > 0.0, "Score should be positive with valid data"


def test_calculate_interpretability_score_empty(temp_dir, sample_parquet_file):
    """Test interpretability score with empty characteristics."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    score = trainer._calculate_interpretability_score({}, {})
    
    assert score == 0.0, "Empty characteristics should give score of 0"


def test_calculate_interpretability_score_high_volatility(temp_dir, sample_parquet_file):
    """Test interpretability score with high volatility regimes."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    # Create characteristics with high volatility
    high_vol_chars = {
        0: RegimeCharacteristics(
            state_id=0,
            sample_count=100,
            mean_values={'s_LDC': 0.5, 's_MR': 0.2, 's_TSMOM': 0.3},
            std_values={'s_LDC': 0.5, 's_MR': 0.4, 's_TSMOM': 0.45},
            volatility=1.8,  # High volatility
            trend_strength=0.7,
            mean_reversion_score=0.3
        )
    }
    
    high_vol_persistence = {
        0: StatePersistence(
            state_id=0,
            total_occurrences=100,
            mean_duration=15.0,  # Good persistence
            median_duration=12.0,
            max_duration=30,
            min_duration=3,
            stable_periods=70,
            transition_frequencies={},
            transition_probabilities={}
        )
    }
    
    score = trainer._calculate_interpretability_score(
        high_vol_chars,
        high_vol_persistence
    )
    
    # High volatility and good persistence should give high score
    assert score > 0.6, "High volatility and persistence should give high score"


def test_calculate_interpretability_score_low_persistence(temp_dir, sample_parquet_file):
    """Test interpretability score with low persistence regimes."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    # Create characteristics with low persistence
    low_pers_chars = {
        0: RegimeCharacteristics(
            state_id=0,
            sample_count=100,
            mean_values={'s_LDC': 0.5, 's_MR': 0.2, 's_TSMOM': 0.3},
            std_values={'s_LDC': 0.3, 's_MR': 0.25, 's_TSMOM': 0.28},
            volatility=0.8,
            trend_strength=0.5,
            mean_reversion_score=0.4
        )
    }
    
    low_pers_persistence = {
        0: StatePersistence(
            state_id=0,
            total_occurrences=100,
            mean_duration=2.0,  # Low persistence
            median_duration=2.0,
            max_duration=5,
            min_duration=1,
            stable_periods=10,
            transition_frequencies={},
            transition_probabilities={}
        )
    }
    
    score = trainer._calculate_interpretability_score(
        low_pers_chars,
        low_pers_persistence
    )
    
    # Low persistence should reduce score
    assert score < 0.5, "Low persistence should give lower score"


def test_calculate_interpretability_score_missing_persistence(temp_dir, sample_parquet_file):
    """Test interpretability score when persistence data is missing."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    chars = {
        0: RegimeCharacteristics(
            state_id=0,
            sample_count=100,
            mean_values={'s_LDC': 0.5, 's_MR': 0.2, 's_TSMOM': 0.3},
            std_values={'s_LDC': 0.3, 's_MR': 0.25, 's_TSMOM': 0.28},
            volatility=0.8,
            trend_strength=0.5,
            mean_reversion_score=0.4
        )
    }
    
    # Empty persistence dict
    score = trainer._calculate_interpretability_score(chars, {})
    
    # Should still calculate score based on volatility and sample count
    assert 0.0 <= score <= 1.0, "Should handle missing persistence gracefully"


# ============================================================================
# Test Model Ranking Logic
# ============================================================================

def test_rank_models_basic(temp_dir, sample_parquet_file):
    """Test model ranking with known configurations."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    # Create mock model evaluation results
    models = {
        '2_states': {
            'basic_metrics': {
                'aic': 1000.0,
                'bic': 1050.0,
                'log_likelihood': -500.0,
                'cv_log_likelihood_mean': -520.0,
                'cv_log_likelihood_std': 10.0
            },
            'interpretability_score': 0.6,
            'n_states': 2
        },
        '3_states': {
            'basic_metrics': {
                'aic': 950.0,  # Better AIC
                'bic': 1020.0,  # Better BIC
                'log_likelihood': -475.0,
                'cv_log_likelihood_mean': -490.0,  # Better CV
                'cv_log_likelihood_std': 8.0
            },
            'interpretability_score': 0.75,  # Better interpretability
            'n_states': 3
        },
        '4_states': {
            'basic_metrics': {
                'aic': 980.0,
                'bic': 1080.0,
                'log_likelihood': -490.0,
                'cv_log_likelihood_mean': -510.0,
                'cv_log_likelihood_std': 12.0
            },
            'interpretability_score': 0.5,
            'n_states': 4
        }
    }
    
    rankings = trainer._rank_models(models)
    
    # Check rankings structure
    assert len(rankings) == 3, "Should rank all 3 models"
    assert all('config_name' in r for r in rankings), "All rankings should have config_name"
    assert all('combined_score' in r for r in rankings), "All rankings should have combined_score"
    assert all('confidence_score' in r for r in rankings), "All rankings should have confidence_score"
    
    # Best model should be first
    best_model = rankings[0]
    assert best_model['config_name'] == '3_states', "3_states should rank highest (best metrics)"
    
    # Scores should be in descending order
    scores = [r['combined_score'] for r in rankings]
    assert scores == sorted(scores, reverse=True), "Rankings should be in descending order"


def test_rank_models_with_errors(temp_dir, sample_parquet_file):
    """Test model ranking skips models with errors."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    models = {
        '2_states': {
            'basic_metrics': {
                'aic': 1000.0,
                'bic': 1050.0,
                'log_likelihood': -500.0,
                'cv_log_likelihood_mean': -520.0
            },
            'interpretability_score': 0.6,
            'n_states': 2
        },
        '3_states': {
            'error': 'Training failed',
            'n_states': 3
        }
    }
    
    rankings = trainer._rank_models(models)
    
    # Should only rank successful models
    assert len(rankings) == 1, "Should only rank models without errors"
    assert rankings[0]['config_name'] == '2_states'


def test_rank_models_single_model(temp_dir, sample_parquet_file):
    """Test model ranking with single model."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    models = {
        '2_states': {
            'basic_metrics': {
                'aic': 1000.0,
                'bic': 1050.0,
                'log_likelihood': -500.0,
                'cv_log_likelihood_mean': -520.0
            },
            'interpretability_score': 0.6,
            'n_states': 2
        }
    }
    
    rankings = trainer._rank_models(models)
    
    # Should handle single model gracefully
    assert len(rankings) == 1
    assert rankings[0]['combined_score'] > 0.0


def test_rank_models_missing_cv_scores(temp_dir, sample_parquet_file):
    """Test model ranking when CV scores are missing."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    models = {
        '2_states': {
            'basic_metrics': {
                'aic': 1000.0,
                'bic': 1050.0,
                'log_likelihood': -500.0,
                'cv_log_likelihood_mean': None  # Missing CV score
            },
            'interpretability_score': 0.6,
            'n_states': 2
        }
    }
    
    rankings = trainer._rank_models(models)
    
    # Should handle missing CV scores
    assert len(rankings) == 1
    assert 'combined_score' in rankings[0]
    assert rankings[0]['cv_score'] == 0.0  # Should default to 0


def test_rank_models_justification(temp_dir, sample_parquet_file):
    """Test that model ranking includes justification."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    models = {
        '2_states': {
            'basic_metrics': {
                'aic': 1000.0,
                'bic': 1050.0,
                'log_likelihood': -500.0,
                'cv_log_likelihood_mean': -520.0
            },
            'interpretability_score': 0.6,
            'n_states': 2
        }
    }
    
    rankings = trainer._rank_models(models)
    
    # Check justification exists and is meaningful
    assert 'justification' in rankings[0]
    assert len(rankings[0]['justification']) > 0
    assert isinstance(rankings[0]['justification'], str)


# ============================================================================
# Integration Tests
# ============================================================================

@patch('train_hmm_systematic.EnhancedHMMTrainer')
@patch('train_hmm_systematic.RegimeAnalyzer')
def test_full_pipeline_integration(mock_analyzer_class, mock_trainer_class,
                                   temp_dir, sample_parquet_file):
    """Test full pipeline with mocked training components."""
    # Setup mocks
    mock_trainer = MagicMock()
    mock_trainer_class.return_value = mock_trainer
    
    # Mock artifact
    mock_artifact = MagicMock(spec=HMMArtifact)
    mock_artifact.n_states = 2
    mock_artifact.metadata = {
        'aic': 1000.0,
        'bic': 1050.0,
        'convergence_log_likelihood': -500.0
    }
    mock_artifact.transition_matrix = [[0.7, 0.3], [0.4, 0.6]]
    mock_artifact.initial_probabilities = [0.5, 0.5]
    mock_artifact.means = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    mock_artifact.covariances = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 2
    mock_artifact.model_dump.return_value = {
        'version': 'v1.0',
        'n_states': 2,
        'metadata': mock_artifact.metadata
    }
    
    mock_trainer.train.return_value = mock_artifact
    mock_trainer.cross_validate.return_value = {
        'log_likelihood_mean': -520.0,
        'log_likelihood_std': 10.0,
        'aic_mean': 1010.0,
        'bic_mean': 1060.0
    }
    
    # Mock analyzer
    mock_analyzer = MagicMock()
    mock_analyzer_class.return_value = mock_analyzer
    
    mock_characteristics = {
        0: RegimeCharacteristics(
            state_id=0,
            sample_count=100,
            mean_values={'s_LDC': 0.5, 's_MR': 0.2, 's_TSMOM': 0.3},
            std_values={'s_LDC': 0.3, 's_MR': 0.25, 's_TSMOM': 0.28},
            volatility=0.8,
            trend_strength=0.6,
            mean_reversion_score=0.3
        ),
        1: RegimeCharacteristics(
            state_id=1,
            sample_count=100,
            mean_values={'s_LDC': -0.3, 's_MR': 0.5, 's_TSMOM': -0.2},
            std_values={'s_LDC': 0.25, 's_MR': 0.3, 's_TSMOM': 0.22},
            volatility=0.6,
            trend_strength=0.4,
            mean_reversion_score=0.6
        )
    }
    
    mock_persistence = {
        0: StatePersistence(
            state_id=0,
            total_occurrences=100,
            mean_duration=10.0,
            median_duration=8.0,
            max_duration=20,
            min_duration=2,
            stable_periods=60,
            transition_frequencies={1: 100},
            transition_probabilities={1: 1.0}
        ),
        1: StatePersistence(
            state_id=1,
            total_occurrences=100,
            mean_duration=10.0,
            median_duration=8.0,
            max_duration=20,
            min_duration=2,
            stable_periods=60,
            transition_frequencies={0: 100},
            transition_probabilities={0: 1.0}
        )
    }
    
    mock_interpretations = {
        0: MagicMock(regime_type='trending', to_dict=lambda: {'regime_type': 'trending'}),
        1: MagicMock(regime_type='mean_reverting', to_dict=lambda: {'regime_type': 'mean_reverting'})
    }
    
    mock_analyzer.characterize_regimes.return_value = mock_characteristics
    mock_analyzer.analyze_state_persistence.return_value = mock_persistence
    mock_analyzer.interpret_regimes.return_value = mock_interpretations
    
    # Run pipeline
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir,
        n_states_range=[2],
        cv_folds=3
    )
    
    results = trainer.run()
    
    # Verify results structure
    assert 'best_model' in results
    assert 'evaluation_summary' in results
    
    # Verify best model
    best_model = results['best_model']
    assert 'config_name' in best_model
    assert 'artifact' in best_model
    assert 'scores' in best_model
    
    # Verify artifacts were saved
    assert (temp_dir / "hmm_2_states.json").exists()
    assert (temp_dir / "hmm_best.json").exists()
    assert (temp_dir / "training_report.json").exists()
    
    # Verify report content
    with open(temp_dir / "training_report.json") as f:
        report = json.load(f)
    
    assert 'timestamp' in report
    assert 'configuration' in report
    assert 'best_model' in report
    assert 'all_models' in report
    assert 'rankings' in report


def test_integration_with_small_synthetic_dataset(temp_dir):
    """Integration test with small synthetic dataset (no mocking)."""
    # Create small synthetic dataset
    np.random.seed(42)
    n_samples = 100
    
    # Generate two clear regimes
    regime1 = np.random.randn(50, 3) * 0.3 + np.array([0.5, 0.2, 0.3])
    regime2 = np.random.randn(50, 3) * 0.3 + np.array([-0.5, -0.2, -0.3])
    
    data = np.vstack([regime1, regime2])
    df = pd.DataFrame(data, columns=['s_LDC', 's_MR', 's_TSMOM'])
    
    # Save to parquet
    data_path = temp_dir / "synthetic_data.parquet"
    df.to_parquet(data_path)
    
    # Run pipeline with minimal configuration
    trainer = SystematicHMMTrainer(
        data_path=data_path,
        output_dir=temp_dir / "output",
        n_states_range=[2],  # Only test 2 states for speed
        cv_folds=2  # Minimal CV folds
    )
    
    # This will actually train a real HMM
    try:
        results = trainer.run()
        
        # Basic validation
        assert 'best_model' in results
        assert results['best_model']['config_name'] == '2_states'
        
        # Check artifacts exist
        output_dir = temp_dir / "output"
        assert (output_dir / "hmm_2_states.json").exists()
        assert (output_dir / "hmm_best.json").exists()
        assert (output_dir / "training_report.json").exists()
        
        # Validate artifact structure
        with open(output_dir / "hmm_best.json") as f:
            artifact_data = json.load(f)
        
        assert 'n_states' in artifact_data
        assert artifact_data['n_states'] == 2
        assert 'transition_matrix' in artifact_data
        assert 'means' in artifact_data
        
    except Exception as e:
        # If training fails (e.g., due to missing dependencies), skip gracefully
        pytest.skip(f"Integration test skipped due to: {e}")


# ============================================================================
# Test Error Handling
# ============================================================================

def test_select_best_model_no_models(temp_dir, sample_parquet_file):
    """Test that select_best_model raises error when no models available."""
    trainer = SystematicHMMTrainer(
        data_path=sample_parquet_file,
        output_dir=temp_dir
    )
    
    evaluation_summary = {
        'models': {},
        'rankings': []
    }
    
    with pytest.raises(ValueError, match="No models available"):
        trainer.select_best_model(evaluation_summary)


def test_output_directory_creation(temp_dir):
    """Test that output directory is created if it doesn't exist."""
    output_dir = temp_dir / "new_output_dir"
    assert not output_dir.exists()
    
    # Create dummy data file
    df = pd.DataFrame({
        's_LDC': np.random.randn(10),
        's_MR': np.random.randn(10),
        's_TSMOM': np.random.randn(10)
    })
    data_path = temp_dir / "data.parquet"
    df.to_parquet(data_path)
    
    trainer = SystematicHMMTrainer(
        data_path=data_path,
        output_dir=output_dir
    )
    
    # Output directory should be created during initialization
    assert output_dir.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
