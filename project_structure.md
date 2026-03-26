# Project Structure

```
wifi_activity_recognition/
├── README.md
├── DEVELOPMENT_GUIDE.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── 
├── wifi_activity_recognition/           # Main package
│   ├── __init__.py
│   ├── version.py
│   ├── cli.py                          # Command line interface
│   ├── 
│   ├── hardware/                       # Hardware abstraction layer
│   │   ├── __init__.py
│   │   ├── base.py                     # Base CSI reader interface
│   │   ├── intel5300.py               # Intel 5300 NIC driver
│   │   ├── esp32.py                   # ESP32 CSI interface
│   │   ├── atheros.py                 # Atheros chipset support
│   │   ├── qualcomm.py                # Qualcomm platform
│   │   ├── broadcom.py                # Planned / inactive support
│   │   ├── mediatek.py                # Planned / inactive support
│   │   └── factory.py                 # Hardware factory pattern
│   │
│   ├── preprocessing/                  # Data preprocessing
│   │   ├── __init__.py
│   │   ├── normalization.py           # CSI normalization
│   │   ├── filtering.py               # Noise filtering
│   │   ├── calibration.py             # Hardware calibration
│   │   ├── segmentation.py            # Time window segmentation
│   │   └── augmentation.py            # Data augmentation
│   │
│   ├── features/                       # Feature extraction
│   │   ├── __init__.py
│   │   ├── time_domain.py             # Statistical features
│   │   ├── frequency_domain.py        # Spectral features
│   │   ├── cv_transforms.py           # Computer vision transforms
│   │   ├── correlation.py             # Cross-correlation features
│   │   └── doppler.py                 # Doppler shift analysis
│   │
│   ├── models/                         # ML models
│   │   ├── __init__.py
│   │   ├── base.py                    # Base model interface
│   │   ├── cnn2d.py                   # 2D CNN models
│   │   ├── cnn3d.py                   # 3D CNN models
│   │   ├── resnet.py                  # ResNet variants
│   │   ├── transformer.py             # Transformer models
│   │   ├── ensemble.py                # Ensemble methods
│   │   ├── factory.py                 # Model factory
│   │   └── serialization.py           # Model artifact helpers
│   │
│   ├── training/                       # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Main training class
│   │   ├── losses.py                  # Custom loss functions
│   │   ├── metrics.py                 # Evaluation metrics
│   │   ├── callbacks.py               # Training callbacks
│   │   └── federated/                 # Federated learning helpers
│   │
│   ├── datasets/                       # Dataset handling
│   │   ├── __init__.py
│   │   ├── loaders.py                 # Data loading utilities
│   │   ├── public_datasets.py         # Public dataset loaders
│   │   ├── synthetic.py               # Synthetic data generation
│   │   └── transforms.py              # Dataset transforms
│   │
│   ├── inference/                      # Real-time inference
│   │   ├── __init__.py
│   │   ├── predictor.py               # Activity predictor
│   │   ├── streaming.py               # Real-time streaming
│   │   ├── postprocessing.py          # Output smoothing
│   │   ├── streaming_pipeline.py      # Streaming orchestration
│   │   └── latency_optimization.py    # Inference optimization
│   │
│   ├── utils/                          # Utilities
│   │   ├── __init__.py
│   │   ├── config.py                  # Configuration management
│   │   ├── logging.py                 # Logging setup
│   │   ├── visualization.py           # Plotting and visualization
│   │   ├── io.py                      # File I/O utilities
│   │   └── performance_monitoring.py  # Runtime monitoring
│   │
│   └── configs/                        # Configuration files
│       ├── hardware/
│       │   ├── intel5300.yaml
│       │   ├── esp32.yaml
│       │   └── default.yaml
│       ├── models/
│       │   ├── cnn2d_default.yaml
│       │   ├── resnet_config.yaml
│       │   └── transformer_config.yaml
│       └── training/
│           ├── default_training.yaml
│           └── quick_training.yaml
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Test configuration
│   ├── benchmarks/
│   ├── cli/
│   ├── datasets/
│   ├── deployment/
│   ├── e2e/
│   ├── features/
│   ├── functional/
│   ├── hardware/
│   ├── inference/
│   ├── models/
│   ├── multimodal/
│   ├── preprocessing/
│   ├── regression/
│   ├── research/
│   ├── training/
│   ├── unit/
│   └── utils/
│
├── examples/                           # Usage examples
│   ├── notebooks/
│   │   ├── 01_getting_started.ipynb
│   │   ├── 02_custom_training.ipynb
│   │   ├── 03_real_time_inference.ipynb
│   │   └── 04_hardware_setup.ipynb
│   ├── scripts/
│   │   ├── train_custom_model.py
│   │   ├── evaluate_model.py
│   │   ├── real_time_demo.py
│   │   └── benchmark_hardware.py
│   └── data/
│       └── sample_datasets/
│
├── docs/                               # Documentation
│   ├── conf.py                        # Sphinx configuration
│   ├── index.rst
│   ├── installation.md
│   ├── quickstart.md
│   ├── api_reference.md
│   ├── hardware_setup.md
│   ├── quickstart.md
│   ├── roadmap_execution_plan.md
│   └── training_guide.md
│
├── benchmarks/                         # Performance benchmarks
│   ├── __init__.py
│   ├── accuracy_benchmark.py
│   ├── latency_benchmark.py
│   ├── memory_benchmark.py
│   └── results/
│
├── deployment/                         # Deployment configurations
│   ├── docker/
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── requirements.txt
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── cloud/
│   │   ├── aws/
│   │   ├── gcp/
│   │   └── azure/
│   └── edge/
│       ├── raspberry_pi/
│       └── jetson/
│
├── .github/                            # GitHub workflows
│   ├── workflows/
│   │   ├── ci.yml                     # Continuous integration
│   │   ├── cd.yml                     # Continuous deployment
│   │   ├── docs.yml                   # Documentation build
│   │   └── benchmarks.yml             # Performance testing
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── hardware_support.md
│   └── PULL_REQUEST_TEMPLATE.md
│
└── scripts/                            # Development scripts
    ├── setup_dev.sh                   # Development environment setup
    ├── run_tests.sh                   # Test runner
    ├── build_docs.sh                  # Documentation builder
    ├── download_models.py             # Pre-trained model downloader
    └── create_release.py              # Release automation
```

## Key Files Breakdown

### Core Package Files

- `__init__.py` - Package initialization and public API
- `version.py` - Version information
- `cli.py` - Command-line interface for common tasks

### Hardware Layer

- `base.py` - Abstract base class for all hardware drivers
- Individual hardware drivers with standardized interfaces
- `factory.py` - Factory pattern for hardware instantiation

### Processing Pipeline

- Clear separation between preprocessing, feature extraction, and modeling
- Each module handles specific aspect of the pipeline
- Modular design for easy extension

### Configuration Management

- YAML-based configuration for hardware, models, and training
- Environment-specific configurations
- Easy customization and experimentation

### Testing Strategy

- Unit tests for individual components
- Integration tests for end-to-end workflows
- Hardware-specific test suites
- Mock data for CI/CD testing

### Documentation Structure

- API documentation with Sphinx
- Tutorials and guides in Markdown
- Jupyter notebooks for interactive examples
- Architecture diagrams and visualizations

This structure provides a solid foundation that's:

- **Modular**: Easy to extend with new hardware/models
- **Testable**: Comprehensive test coverage
- **Documented**: Clear documentation structure
- **Deployable**: Ready for various deployment scenarios
- **Professional**: Follows Python packaging best practices

