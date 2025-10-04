# HMM Research Environment Documentation

Complete documentation for the Hidden Markov Model Research Environment.

## 📚 Documentation Index

### Getting Started

- **[Getting Started Tutorial](../notebooks/00_getting_started_tutorial.ipynb)** - Start here! Complete introduction with hands-on examples
- **[Quick Reference](QUICK_REFERENCE.md)** - Quick command reference and common patterns
- **[Development Setup](DEVELOPMENT_SETUP.md)** - Environment setup and installation

### Core Documentation

- **[API Documentation](HMM_RESEARCH_API.md)** - Comprehensive API reference for all components
- **[Best Practices](BEST_PRACTICES.md)** - Recommended practices for research and production
- **[Example Configurations](EXAMPLE_CONFIGURATIONS.md)** - Pre-configured setups for different scenarios
- **[Integration Examples](INTEGRATION_EXAMPLES.md)** - End-to-end workflow examples

### Troubleshooting & Support

- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Solutions to common issues
- **[Development Workflow](DEVELOPMENT_WORKFLOW.md)** - Development best practices

### Tutorials

- **[Production Deployment Tutorial](../notebooks/06_production_deployment_tutorial.ipynb)** - Complete deployment guide
- **[Research Notebooks](../notebooks/README.md)** - Interactive Jupyter notebooks

---

## 🚀 Quick Start

### For First-Time Users

1. **Install dependencies:**
   ```bash
   cd py
   pip install -e ".[research]"
   ```

2. **Start with the tutorial:**
   ```bash
   jupyter notebook ../notebooks/00_getting_started_tutorial.ipynb
   ```

3. **Read the API documentation:**
   - [HMM Training](HMM_RESEARCH_API.md#hmm-training)
   - [Regime Analysis](HMM_RESEARCH_API.md#regime-analysis)
   - [Visualization](HMM_RESEARCH_API.md#visualization)

### For Experienced Users

- **API Reference:** [HMM_RESEARCH_API.md](HMM_RESEARCH_API.md)
- **Best Practices:** [BEST_PRACTICES.md](BEST_PRACTICES.md)
- **Example Configs:** [EXAMPLE_CONFIGURATIONS.md](EXAMPLE_CONFIGURATIONS.md)

---

## 📖 Documentation by Topic

### Data Management

**Loading Data:**
- [LDC Signal Integration](INTEGRATION_EXAMPLES.md#ldc-signal-integration)
- [Data Validation](HMM_RESEARCH_API.md#data-integration)
- [Preprocessing](BEST_PRACTICES.md#data-preparation)

**Quality Assurance:**
- [Data Quality Checks](BEST_PRACTICES.md#always-validate-data-quality)
- [Handling Missing Values](BEST_PRACTICES.md#handle-missing-values-appropriately)
- [Outlier Detection](TROUBLESHOOTING.md#data-loading-issues)

### Model Training

**Basic Training:**
- [EnhancedHMMTrainer API](HMM_RESEARCH_API.md#enhancedhmmtrainer)
- [Training Best Practices](BEST_PRACTICES.md#model-training)
- [Multi-Library Support](INTEGRATION_EXAMPLES.md#multi-library-comparison)

**Advanced Training:**
- [Hyperparameter Tuning](INTEGRATION_EXAMPLES.md#automated-hyperparameter-tuning)
- [Cross-Validation](BEST_PRACTICES.md#use-cross-validation)
- [Model Comparison](HMM_RESEARCH_API.md#model-evaluation)

**Troubleshooting:**
- [Convergence Issues](TROUBLESHOOTING.md#model-fails-to-converge)
- [Numerical Instability](TROUBLESHOOTING.md#numerical-instability)
- [Poor Performance](TROUBLESHOOTING.md#poor-model-performance)

### Regime Analysis

**Analysis Tools:**
- [RegimeAnalyzer API](HMM_RESEARCH_API.md#regimeanalyzer)
- [Regime Characterization](BEST_PRACTICES.md#analyze-regime-stability)
- [Economic Interpretation](BEST_PRACTICES.md#validate-economic-interpretation)

**Visualization:**
- [RegimeVisualizer API](HMM_RESEARCH_API.md#regimevisualizer)
- [Visualization Best Practices](BEST_PRACTICES.md#visualization)
- [Interactive Dashboards](HMM_RESEARCH_API.md#create_regime_dashboard)

### Model Evaluation

**Evaluation Methods:**
- [Cross-Validation](HMM_RESEARCH_API.md#cross_validate)
- [Model Comparison](HMM_RESEARCH_API.md#compare_models)
- [Performance Metrics](BEST_PRACTICES.md#compare-multiple-metrics)

**Validation:**
- [Out-of-Sample Testing](BEST_PRACTICES.md#test-on-out-of-sample-data)
- [Regime Stability Analysis](BEST_PRACTICES.md#analyze-regime-stability)
- [Statistical Validation](INTEGRATION_EXAMPLES.md#ab-testing-configuration)

### Production Deployment

**Deployment Process:**
- [Production Tutorial](../notebooks/06_production_deployment_tutorial.ipynb)
- [Deployment Pipeline](INTEGRATION_EXAMPLES.md#production-deployment-pipeline)
- [Artifact Management](HMM_RESEARCH_API.md#artifact-management)

**Monitoring:**
- [Performance Monitoring](BEST_PRACTICES.md#monitor-production-performance)
- [Real-Time Inference](INTEGRATION_EXAMPLES.md#real-time-regime-detection)
- [Rollback Procedures](BEST_PRACTICES.md#plan-for-model-updates)

---

## 🎯 Documentation by Use Case

### High-Frequency Trading

**Relevant Documentation:**
- [HFT Configuration](EXAMPLE_CONFIGURATIONS.md#high-frequency-trading-hft)
- [Performance Optimization](EXAMPLE_CONFIGURATIONS.md#performance-optimized-configurations)
- [Real-Time Inference](INTEGRATION_EXAMPLES.md#real-time-regime-detection)

**Key Topics:**
- Minimal latency inference
- Simple model structures
- Fast preprocessing

### Swing Trading

**Relevant Documentation:**
- [Swing Trading Configuration](EXAMPLE_CONFIGURATIONS.md#swing-trading)
- [Regime Analysis](HMM_RESEARCH_API.md#regime-analysis)
- [Model Evaluation](BEST_PRACTICES.md#model-evaluation)

**Key Topics:**
- Medium-term regime detection
- Balanced accuracy
- Comprehensive validation

### Portfolio Management

**Relevant Documentation:**
- [Long-Term Configuration](EXAMPLE_CONFIGURATIONS.md#long-term-portfolio-management)
- [Economic Interpretation](BEST_PRACTICES.md#validate-economic-interpretation)
- [Regime Characterization](HMM_RESEARCH_API.md#calculate_state_statistics)

**Key Topics:**
- Stable regime detection
- Economic interpretation
- Long-term analysis

### Cryptocurrency Trading

**Relevant Documentation:**
- [Crypto Configuration](EXAMPLE_CONFIGURATIONS.md#cryptocurrency-markets)
- [High Volatility Handling](TROUBLESHOOTING.md#numerical-instability)
- [Robust Training](BEST_PRACTICES.md#use-multiple-random-seeds)

**Key Topics:**
- High volatility handling
- Rapid regime changes
- Robust outlier detection

---

## 🔧 Common Tasks

### Train Your First Model

```python
from imp.hmm.trainer import EnhancedHMMTrainer
from imp.data.preprocessor import SignalPreprocessor

# Preprocess data
preprocessor = SignalPreprocessor()
observations, _ = preprocessor.preprocess(data, normalize=True)

# Train model
trainer = EnhancedHMMTrainer(n_states=3, random_state=42)
artifact = trainer.train(observations)
```

**Learn More:** [Getting Started Tutorial](../notebooks/00_getting_started_tutorial.ipynb)

### Analyze Regimes

```python
from imp.hmm.inference import HMMInference
from imp.hmm.regime_analysis import RegimeAnalyzer

# Perform inference
inference = HMMInference(artifact)
state_probs = inference.predict_proba(observations)

# Analyze regimes
analyzer = RegimeAnalyzer(artifact)
analysis = analyzer.analyze_regimes(observations, state_probs)
```

**Learn More:** [Regime Analysis API](HMM_RESEARCH_API.md#regime-analysis)

### Visualize Results

```python
from imp.visualization.regime_visualizer import RegimeVisualizer

visualizer = RegimeVisualizer(artifact)
fig = visualizer.plot_state_probabilities(state_probs)
fig.show()
```

**Learn More:** [Visualization API](HMM_RESEARCH_API.md#visualization)

### Deploy to Production

```python
from imp.hmm.artifact_management import ArtifactManager

manager = ArtifactManager()
manager.save_artifact(
    artifact,
    name='production_model',
    version='1.0.0',
    metadata={'training_date': '2025-01-15'}
)
```

**Learn More:** [Production Deployment Tutorial](../notebooks/06_production_deployment_tutorial.ipynb)

---

## 📊 Example Workflows

### Research Workflow

1. **Data Exploration** → [01_data_exploration.ipynb](../notebooks/01_data_exploration.ipynb)
2. **Model Comparison** → [02_hmm_training_comparison.ipynb](../notebooks/02_hmm_training_comparison.ipynb)
3. **Regime Analysis** → [03_regime_analysis.ipynb](../notebooks/03_regime_analysis.ipynb)
4. **Parameter Tuning** → [04_parameter_optimization.ipynb](../notebooks/04_parameter_optimization.ipynb)

### Production Workflow

1. **Train Model** → [Best Practices](BEST_PRACTICES.md#model-training)
2. **Validate** → [Model Evaluation](BEST_PRACTICES.md#model-evaluation)
3. **Test** → [Production Testing](../notebooks/06_production_deployment_tutorial.ipynb)
4. **Deploy** → [Deployment Pipeline](INTEGRATION_EXAMPLES.md#production-deployment-pipeline)
5. **Monitor** → [Performance Monitoring](BEST_PRACTICES.md#monitor-production-performance)

---

## 🆘 Getting Help

### Common Issues

1. **Installation Problems** → [Troubleshooting: Installation](TROUBLESHOOTING.md#installation-issues)
2. **Training Failures** → [Troubleshooting: Training](TROUBLESHOOTING.md#training-problems)
3. **Performance Issues** → [Troubleshooting: Performance](TROUBLESHOOTING.md#performance-issues)
4. **Visualization Problems** → [Troubleshooting: Visualization](TROUBLESHOOTING.md#visualization-problems)

### Debug Checklist

When encountering issues, check:

- [ ] All dependencies installed
- [ ] Data loaded correctly
- [ ] Data preprocessed (normalized, no NaN/inf)
- [ ] Model parameters reasonable
- [ ] Sufficient training iterations
- [ ] Error messages reviewed

**Full Checklist:** [Troubleshooting Guide](TROUBLESHOOTING.md#debug-checklist)

### Additional Resources

- **Examples:** [py/examples/](../examples/)
- **Tests:** [py/tests/](../tests/)
- **Notebooks:** [notebooks/](../notebooks/)

---

## 📝 Contributing to Documentation

When updating documentation:

1. **Keep it practical** - Focus on actionable information
2. **Include examples** - Show, don't just tell
3. **Link related docs** - Help users navigate
4. **Test code examples** - Ensure they work
5. **Update index** - Keep this README current

---

## 📄 Document Summaries

### [HMM_RESEARCH_API.md](HMM_RESEARCH_API.md)
Complete API reference for all components. Use this for detailed method signatures and parameters.

### [BEST_PRACTICES.md](BEST_PRACTICES.md)
Recommended practices for data preparation, training, evaluation, and deployment. Essential reading for production use.

### [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
Solutions to common problems. Start here when encountering issues.

### [INTEGRATION_EXAMPLES.md](INTEGRATION_EXAMPLES.md)
End-to-end workflow examples. Use these as templates for your own implementations.

### [EXAMPLE_CONFIGURATIONS.md](EXAMPLE_CONFIGURATIONS.md)
Pre-configured setups for different scenarios. Copy and adapt for your use case.

### [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
Quick command reference. Use for fast lookups.

### [DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md)
Environment setup instructions. Follow for initial installation.

### [DEVELOPMENT_WORKFLOW.md](DEVELOPMENT_WORKFLOW.md)
Development best practices. Read when contributing code.

---

## 🔄 Documentation Updates

**Last Updated:** 2025-01-15

**Recent Changes:**
- Added comprehensive API documentation
- Created troubleshooting guide
- Added best practices document
- Created integration examples
- Added example configurations
- Created tutorial notebooks

**Version:** 1.0.0

---

## 📞 Support

For questions or issues:

1. Check [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Review [API Documentation](HMM_RESEARCH_API.md)
3. See [Examples](INTEGRATION_EXAMPLES.md)
4. Review [Best Practices](BEST_PRACTICES.md)

---

**Happy researching! 🚀**
