# AGENTS.md - Coding Agent Instructions

This file provides comprehensive instructions for AI coding agents to develop the WiFi Activity Recognition package. Follow these guidelines to maintain consistency, quality, and architectural integrity.

## 🎯 Project Overview

**Project**: WiFi Activity Recognition Package<br>
**Purpose**: CSI-based human activity recognition using computer vision<br>
**Architecture**: Modular, hardware-agnostic, production-ready Python package<br>
**Target Users**: Researchers, developers, IoT practitioners

## 📋 Development Priorities

### Phase 1 (Current): Core Implementation

1. **Hardware Drivers** (Intel 5300, ESP32)
2. **Basic Models** (CNN2D, ResNet variants)
3. **Preprocessing Pipeline** (normalization, filtering)
4. **Training Framework** (PyTorch-based)
5. **Unit Testing** (>90% coverage goal)

### Phase 2: Advanced Features

1. **Additional Hardware** (Atheros, Qualcomm)
2. **Advanced Models** (3D CNN, Transformers)
3. **Real-time Inference** (streaming pipeline)
4. **Deployment Tools** (Docker, edge optimization)

## 🏗️ Architecture Principles

### Core Design Patterns

- **Hardware Abstraction**: All hardware specifics isolated behind `CSIReaderBase`
- **Factory Pattern**: Use `HardwareFactory` for hardware instantiation
- **Standardized Data**: Everything uses `CSIData` format
- **Configuration-Driven**: YAML configs for all parameters
- **Plugin Architecture**: Easy to extend with new components

### Data Flow Architecture

```
Raw Hardware Data → CSIData → Preprocessing → Features → Model → Activity
```

### Module Dependencies

```
cli.py → [hardware, models, training, inference]
hardware/ → base.py (no external deps)
models/ → base.py, preprocessing/
training/ → models/, datasets/
inference/ → models/, preprocessing/
```

## 📁 File Organization Rules

### Directory Structure

```
wifi_activity_recognition/
├── hardware/          # Hardware abstraction layer
├── preprocessing/     # Data processing pipeline
├── features/         # Feature extraction
├── models/           # ML model implementations
├── training/         # Training pipeline
├── datasets/         # Dataset handling
├── inference/        # Real-time inference
├── utils/           # Shared utilities
└── configs/         # Configuration files
```

### File Naming Conventions

- **Classes**: `PascalCase` (e.g., `ESP32Reader`, `CNN2DModel`)
- **Functions**: `snake_case` (e.g., `load_csi_data`, `normalize_amplitude`)
- **Files**: `snake_case.py` (e.g., `intel5300.py`, `cnn_models.py`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_SAMPLING_RATE`)

## 🔧 Implementation Guidelines

### Hardware Driver Development

**When implementing hardware drivers (e.g., `intel5300.py`, `esp32.py`):**

```python
# Template structure for hardware drivers
from .base import CSIReaderBase, CSIData, HardwareConfig
import numpy as np

class NewHardwareReader(CSIReaderBase):
    """Driver for [Hardware Name] CSI extraction."""

    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        # Hardware-specific initialization

    def connect(self) -> bool:
        # Establish hardware connection
        # Set self._is_connected = True on success

    def read_packet(self) -> Optional[CSIData]:
        # Read raw data from hardware
        # Convert to standardized CSIData format
        # Apply hardware-specific calibration

    def get_hardware_info(self) -> Dict[str, Any]:
        # Return hardware specifications
```

**Key Requirements:**

- Always inherit from `CSIReaderBase`
- Convert all data to `CSIData` format
- Handle hardware-specific calibration internally
- Implement proper error handling and timeouts
- Add comprehensive docstrings with hardware details

### Model Implementation

**When implementing models (e.g., `cnn2d.py`, `resnet.py`):**

```python
# Template for model implementations
import torch
import torch.nn as nn
from .base import BaseActivityModel

class CNN2DModel(BaseActivityModel):
    """2D CNN for activity recognition from CSI spectrograms."""

    def __init__(self, num_classes: int, input_shape: tuple):
        super().__init__(num_classes, input_shape)
        # Define network architecture

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass implementation

    def get_model_info(self) -> Dict[str, Any]:
        # Return model specifications
```

**Key Requirements:**

- Always inherit from `BaseActivityModel`
- Handle variable input sizes (different hardware = different CSI shapes)
- Use adaptive pooling for hardware compatibility
- Include model metadata and configuration
- Implement both PyTorch and TensorFlow variants when possible

### Preprocessing Implementation

**When implementing preprocessing (e.g., `normalization.py`, `filtering.py`):**

```python
# Template for preprocessing modules
from ..hardware.base import CSIData
import numpy as np

def normalize_csi_data(csi_data: CSIData, method: str = "minmax") -> CSIData:
    """
    Normalize CSI data using specified method.

    Args:
        csi_data: Input CSI data
        method: Normalization method ("minmax", "zscore", "log")

    Returns:
        Normalized CSI data (new CSIData object)
    """
    # Process amplitude and phase separately
    # Return new CSIData object with processed data
    # Preserve all metadata
```

**Key Requirements:**

- Always work with `CSIData` objects
- Return new objects (immutable operations)
- Support multiple processing methods
- Handle edge cases (zero variance, invalid values)
- Include comprehensive parameter validation

## 📝 Coding Standards

### Code Quality Requirements

**Every file must include:**

```python
"""
Module docstring with:
- Brief description
- Key classes/functions
- Usage examples
- Hardware compatibility notes (if applicable)
"""

from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)
```

**Function Documentation:**

```python
def process_function(param1: Type, param2: Type = default) -> ReturnType:
    """
    Brief description of what the function does.

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 with default

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this exception occurs

    Example:
        >>> result = process_function(value1, value2)
        >>> print(result)
    """
```

### Error Handling Patterns

```python
# Use specific exceptions
class CSIProcessingError(Exception):
    """Raised when CSI processing fails."""
    pass

# Log errors appropriately
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise CSIProcessingError(f"Processing failed: {e}") from e
```

### Configuration Management

```python
# Load configs using utils
from ..utils.config import load_config

def initialize_component(config_path: str = None):
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config("configs/default.yaml")
```

## 🧪 Testing Requirements

### Test Structure

```python
# tests/test_module/test_component.py
import pytest
from unittest.mock import Mock, patch
from wifi_activity_recognition.module import Component

class TestComponent:
    """Test suite for Component class."""

    def setup_method(self):
        """Setup for each test method."""
        self.component = Component()

    def test_basic_functionality(self):
        """Test basic component functionality."""
        # Test implementation

    def test_error_conditions(self):
        """Test error handling."""
        # Test error cases

    @pytest.mark.hardware
    def test_hardware_integration(self):
        """Test with actual hardware (optional)."""
        # Hardware integration tests
```

### Testing Priorities

1. **Unit Tests**: Every public function and method
2. **Integration Tests**: End-to-end workflows
3. **Mock Tests**: Hardware interactions without real hardware
4. **Edge Cases**: Invalid inputs, network failures, etc.
5. **Performance Tests**: Memory usage, processing speed

## 📊 Data and Model Standards

### CSI Data Processing

```python
# Always validate CSI data
from ..hardware.base import validate_csi_data

def process_csi(csi_data: CSIData) -> ProcessedData:
    if not validate_csi_data(csi_data):
        raise ValueError("Invalid CSI data")

    # Process data
    return processed_data
```

### Model Training Patterns

```python
# Standard training loop structure
class Trainer:
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            total_loss += loss.item()

        return total_loss / len(dataloader)
```

## 🔍 Code Review Checklist

Before implementing any component, ensure:

### Functionality

- [ ] Follows the established architecture patterns
- [ ] Uses standardized data formats (`CSIData`)
- [ ] Implements proper error handling
- [ ] Includes comprehensive logging
- [ ] Handles edge cases and invalid inputs

### Code Quality

- [ ] Includes type hints for all functions
- [ ] Has comprehensive docstrings
- [ ] Follows naming conventions
- [ ] Passes all linting checks (black, flake8, mypy)
- [ ] Has >90% test coverage

### Integration

- [ ] Works with existing components
- [ ] Uses configuration management
- [ ] Registers with factory classes (if applicable)
- [ ] Updates CLI commands (if applicable)
- [ ] Updates documentation

### Performance

- [ ] Efficient memory usage
- [ ] Reasonable processing speed
- [ ] Handles large datasets
- [ ] Suitable for real-time applications (when required)

## 🚀 Implementation Workflow

### For New Features

1. **Design**: Review architecture and existing patterns
2. **Interface**: Define classes and function signatures
3. **Implementation**: Write core functionality
4. **Testing**: Create comprehensive test suite
5. **Integration**: Connect with existing components
6. **Documentation**: Update docs and examples
7. **CLI**: Add command-line interface (if needed)

### For Bug Fixes

1. **Reproduce**: Create failing test case
2. **Diagnose**: Identify root cause
3. **Fix**: Implement minimal fix
4. **Test**: Ensure fix works and doesn't break anything
5. **Document**: Update relevant documentation

### For Hardware Support

1. **Research**: Understand hardware CSI format
2. **Driver**: Implement `CSIReaderBase` subclass
3. **Calibration**: Add hardware-specific calibration
4. **Testing**: Test with real hardware (if available)
5. **Mock Testing**: Create mock data for CI/CD
6. **Documentation**: Add hardware setup guide
7. **Registration**: Register with `HardwareFactory`

## 🎛️ Configuration Standards

### YAML Configuration Format

```yaml
# configs/hardware/esp32.yaml
hardware:
  name: "ESP32 CSI"
  type: "esp32"

connection:
  serial_port: "/dev/ttyUSB0"
  baud_rate: 115200
  timeout: 1.0

csi_params:
  sampling_rate: 100
  channel: 6
  bandwidth: 20
  subcarriers: 64

processing:
  calibration_required: false
  amplitude_scaling: "log"
  phase_unwrap: false
```

### Configuration Loading

```python
from ..utils.config import load_config, validate_config

def initialize_hardware(config_path: str):
    config = load_config(config_path)
    validate_config(config, "hardware_schema.yaml")
    return create_hardware_from_config(config)
```

## 🔧 Debugging and Troubleshooting

### Logging Patterns

```python
import logging
logger = logging.getLogger(__name__)

# Different log levels
logger.debug("Detailed debugging info")
logger.info("General information")
logger.warning("Something unexpected happened")
logger.error("An error occurred")
logger.critical("Serious error occurred")
```

### Common Issues and Solutions

**CSI Data Format Issues:**

- Always validate with `validate_csi_data()`
- Check shapes match expected dimensions
- Verify timestamp reasonableness
- Handle NaN/infinite values

**Hardware Connection Issues:**

- Implement proper timeout handling
- Add retry logic for transient failures
- Provide clear error messages
- Include hardware-specific troubleshooting

**Model Training Issues:**

- Check data preprocessing pipeline
- Validate input shapes and data types
- Monitor loss curves and metrics
- Handle class imbalance

**Performance Issues:**

- Profile memory usage
- Optimize critical paths
- Use appropriate data structures
- Consider batch processing

## 📈 Success Metrics

### Code Quality Metrics

- **Test Coverage**: >90%
- **Type Coverage**: 100% for public API
- **Documentation Coverage**: All public functions
- **Linting**: Zero warnings/errors

### Performance Metrics

- **Inference Latency**: <50ms for real-time applications
- **Memory Usage**: <256MB for edge deployment
- **Training Speed**: Reasonable epoch times
- **Accuracy**: Competitive with research benchmarks

### User Experience Metrics

- **Setup Time**: <5 minutes for basic usage
- **Hardware Support**: Clear compatibility matrix
- **Documentation Quality**: Users can follow without external help
- **Error Messages**: Clear, actionable feedback

--------------------------------------------------------------------------------

## 🎯 Final Notes for Agents

**Remember:**

- This is a research-grade package that should also be production-ready
- Hardware compatibility is a key differentiator
- Real-time performance matters for practical applications
- The package should be accessible to both researchers and practitioners
- Code quality and maintainability are as important as functionality

**When in doubt:**

- Follow existing patterns in the codebase
- Prioritize clarity and maintainability over cleverness
- Add comprehensive tests and documentation
- Ask for clarification through GitHub issues

**Success means:**

- A researcher can train custom models easily
- A developer can deploy to production
- A student can learn WiFi sensing concepts
- The community can extend and contribute

Build something that makes WiFi sensing accessible to everyone! 🚀
