# Contributing to WiFi Activity Recognition

Thank you for your interest in contributing to the WiFi Activity Recognition package! We welcome contributions from the community and appreciate your help in making this project better.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## 🤝 Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for all.

## 🎯 How to Contribute

There are many ways to contribute to this project:

### 🐛 Report Bugs
- Use the GitHub issue tracker
- Include detailed reproduction steps
- Provide system information and hardware details
- Include relevant logs and error messages

### 💡 Suggest Features
- Check existing issues first to avoid duplicates
- Describe the feature and its use case
- Explain why it would be valuable to the community
- Consider implementation complexity and maintenance

### 📝 Improve Documentation
- Fix typos and grammatical errors
- Add missing documentation
- Improve existing examples
- Create tutorials and guides

### 🔧 Code Contributions
- Fix bugs and issues
- Implement new features
- Add support for new hardware platforms
- Improve performance and optimization
- Add tests and improve coverage

### 🧪 Hardware Support
- Add drivers for new WiFi hardware
- Test existing drivers on different platforms
- Improve hardware compatibility
- Add hardware-specific optimizations

## 🛠️ Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Hardware for testing (optional but recommended)

### Setting up the Development Environment

1. **Fork and Clone the Repository**
```bash
git clone https://github.com/diogoribeiro7/wifi-csi-activity-recognition.git
cd wifi-csi-activity-recognition
```

2. **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Development Dependencies**
```bash
pip install -e ".[dev]"
```

4. **Install Pre-commit Hooks**
```bash
pre-commit install
```

5. **Run Tests to Verify Setup**
```bash
pytest tests/
```

### Development Workflow

1. **Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make Your Changes**
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

3. **Run Tests and Linting**
```bash
# Run tests
pytest tests/

# Run linting
black wifi_activity_recognition/
flake8 wifi_activity_recognition/
isort wifi_activity_recognition/

# Type checking
mypy wifi_activity_recognition/
```

4. **Commit Your Changes**
```bash
git add .
git commit -m "feat: add support for new hardware platform"
```

5. **Push and Create Pull Request**
```bash
git push origin feature/your-feature-name
```

## 📏 Contribution Guidelines

### Code Style
We follow PEP 8 and use automated tools to enforce consistency:

- **Black** for code formatting (line length: 88 characters)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

### Commit Message Format
We use conventional commit messages:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(hardware): add ESP32 CSI driver support
fix(preprocessing): resolve phase unwrapping edge case
docs(readme): update installation instructions
test(models): add unit tests for CNN2D architecture
```

### Documentation Standards

- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Type Hints**: Include type hints for all function parameters and return values
- **Comments**: Add inline comments for complex logic
- **Examples**: Include usage examples in docstrings and README

Example docstring:
```python
def process_csi_data(csi_data: CSIData, method: str = "normalize") -> np.ndarray:
    """
    Process raw CSI data using specified method.

    Args:
        csi_data: Raw CSI data from hardware
        method: Processing method ("normalize", "filter", "calibrate")

    Returns:
        Processed CSI data as numpy array

    Raises:
        ValueError: If method is not supported

    Example:
        >>> csi = CSIData(...)
        >>> processed = process_csi_data(csi, method="normalize")
    """
```

### Testing Requirements

- **Unit Tests**: Add tests for all new functions and classes
- **Integration Tests**: Add tests for end-to-end workflows
- **Hardware Tests**: Add tests for hardware-specific functionality (when possible)
- **Coverage**: Aim for >90% test coverage
- **Mock Data**: Use mock data for testing when hardware is not available

Test file structure:
```
tests/
├── test_hardware/
│   ├── test_base.py
│   └── test_esp32.py
├── test_models/
│   └── test_cnn2d.py
└── test_integration/
    └── test_end_to_end.py
```

### Performance Considerations

- **Memory Efficiency**: Optimize memory usage for large datasets
- **Processing Speed**: Profile and optimize critical paths
- **Real-time Performance**: Ensure low-latency for streaming applications
- **Hardware Resources**: Consider resource constraints on edge devices

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure all tests pass**
```bash
pytest tests/
```

2. **Check code quality**
```bash
pre-commit run --all-files
```

3. **Update documentation**
- Update README if adding new features
- Update API documentation
- Add usage examples

4. **Update CHANGELOG**
- Add entry describing your changes
- Follow the existing format

### Pull Request Template

When creating a pull request, please include:

- **Description**: Clear description of changes and motivation
- **Type of Change**: Bug fix, new feature, documentation, etc.
- **Testing**: How you tested your changes
- **Checklist**: Confirm you've followed the guidelines
- **Screenshots**: If applicable, for UI/visualization changes
- **Hardware Tested**: List any hardware platforms tested

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Code Review**: Maintainers and community members review the code
3. **Discussion**: Address feedback and questions
4. **Approval**: Once approved, the PR will be merged

## 🐛 Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Environment Information**:
  - Operating system
  - Python version
  - Package version
  - Hardware platform (if applicable)

- **Steps to Reproduce**:
  - Minimal code example
  - Expected behavior
  - Actual behavior
  - Error messages and stack traces

- **Additional Context**:
  - Screenshots (if applicable)
  - Configuration files
  - Log files

### Feature Requests

For feature requests, please include:

- **Use Case**: Describe the problem you're trying to solve
- **Proposed Solution**: Your idea for how to address it
- **Alternatives**: Other solutions you've considered
- **Implementation**: Thoughts on how it might be implemented

## 💬 Community

### Communication Channels

- **GitHub Discussions**: For general questions and community discussion
- **GitHub Issues**: For bug reports and feature requests
- **Email**: For private inquiries and collaboration opportunities

### Getting Help

If you need help with:
- **Development Setup**: Check the documentation or open a discussion
- **Hardware Configuration**: Refer to hardware-specific guides
- **Algorithm Questions**: Join the research discussions
- **Performance Issues**: Create an issue with profiling data

### Recognition

We appreciate all contributions! Contributors will be:
- Listed in the AUTHORS.md file
- Mentioned in release notes for significant contributions
- Invited to join the core team for sustained contributions

## 📚 Resources

### Useful Links
- [Project Documentation](https://wifi-activity-recognition.readthedocs.io/)
- [API Reference](https://wifi-activity-recognition.readthedocs.io/en/latest/api/)
- [Hardware Setup Guides](docs/hardware_setup.md)
- [Model Training Tutorials](docs/training_guide.md)

### Research Papers
- [Key papers and references](docs/references.md)
- [Background on WiFi sensing](docs/background.md)

### External Resources
- [WiFi CSI Research Community](https://github.com/topics/wifi-csi)
- [Computer Vision Resources](https://github.com/topics/computer-vision)

---

Thank you for contributing to WiFi Activity Recognition! 🚀

For questions about contributing, please open a discussion on GitHub or contact the maintainers.
