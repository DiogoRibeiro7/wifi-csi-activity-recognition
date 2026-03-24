#!/bin/bash

# WiFi Activity Recognition - Development Environment Setup Script
# This script sets up the complete development environment for contributors

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="wifi-activity-recognition"
PYTHON_MIN_VERSION="3.8"
VENV_NAME="wifi-har-dev"

# Functions
print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_python_version() {
    if command -v python3 >/dev/null 2>&1; then
        local version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        local required="3.8"
        if [ "$(printf '%s\n' "$required" "$version" | sort -V | head -n1)" = "$required" ]; then
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}

# Main setup function
main() {
    print_header "WiFi Activity Recognition - Development Setup"
    
    echo -e "${PURPLE}Setting up the development environment...${NC}\n"
    
    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ] || [ ! -f "DEVELOPMENT_GUIDE.md" ]; then
        print_error "Please run this script from the project root directory"
        print_info "Make sure you have pyproject.toml and DEVELOPMENT_GUIDE.md in the current directory"
        exit 1
    fi
    
    print_success "Found project files in current directory"
    
    # System requirements check
    print_header "Checking System Requirements"
    
    # Check Python
    if check_python_version; then
        local version=$(python3 --version)
        print_success "Python found: $version"
    else
        print_error "Python 3.8+ is required but not found"
        print_info "Please install Python 3.8 or higher:"
        print_info "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        print_info "  macOS: brew install python3"
        print_info "  Windows: Download from python.org"
        exit 1
    fi
    
    # Check Git
    if check_command git; then
        print_success "Git found: $(git --version)"
    else
        print_error "Git is required but not found"
        print_info "Please install Git:"
        print_info "  Ubuntu/Debian: sudo apt install git"
        print_info "  macOS: brew install git"
        print_info "  Windows: Download from git-scm.com"
        exit 1
    fi
    
    # Check optional but recommended tools
    print_info "Checking optional development tools..."
    
    if check_command make; then
        print_success "Make found"
    else
        print_warning "Make not found (optional for build automation)"
    fi
    
    if check_command docker; then
        print_success "Docker found"
    else
        print_warning "Docker not found (optional for containerized development)"
    fi
    
    # Virtual Environment Setup
    print_header "Setting Up Virtual Environment"
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment '$VENV_NAME' already exists"
        read -p "Remove and recreate? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
            print_info "Removed existing virtual environment"
        else
            print_info "Using existing virtual environment"
        fi
    fi
    
    if [ ! -d "$VENV_NAME" ]; then
        print_info "Creating virtual environment '$VENV_NAME'..."
        python3 -m venv "$VENV_NAME"
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source "$VENV_NAME/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    print_success "Pip upgraded"
    
    # Install package in development mode
    print_header "Installing Package Dependencies"
    
    print_info "Installing package in development mode..."
    pip install -e ".[dev]"
    print_success "Package installed in development mode"
    
    # Install additional development tools
    print_info "Installing additional development tools..."
    
    # Tools for debugging and profiling
    pip install ipdb line_profiler memory_profiler
    
    # Tools for documentation
    pip install sphinx-autobuild
    
    # Tools for testing
    pip install pytest-xdist pytest-benchmark pytest-mock
    
    print_success "Additional tools installed"
    
    # Pre-commit setup
    print_header "Setting Up Code Quality Tools"
    
    if [ -f ".pre-commit-config.yaml" ]; then
        print_info "Installing pre-commit hooks..."
        pre-commit install
        print_success "Pre-commit hooks installed"
        
        print_info "Running pre-commit on all files..."
        pre-commit run --all-files || print_warning "Some pre-commit checks failed (this is normal for initial setup)"
    else
        print_warning ".pre-commit-config.yaml not found, skipping pre-commit setup"
    fi
    
    # Create necessary directories
    print_header "Creating Project Structure"
    
    directories=(
        "wifi_activity_recognition/hardware"
        "wifi_activity_recognition/preprocessing"
        "wifi_activity_recognition/features"
        "wifi_activity_recognition/models"
        "wifi_activity_recognition/training"
        "wifi_activity_recognition/datasets"
        "wifi_activity_recognition/inference"
        "wifi_activity_recognition/utils"
        "wifi_activity_recognition/configs/hardware"
        "wifi_activity_recognition/configs/models"
        "wifi_activity_recognition/configs/training"
        "tests/test_hardware"
        "tests/test_preprocessing"
        "tests/test_models"
        "tests/test_training"
        "tests/test_integration"
        "tests/data"
        "examples/notebooks"
        "examples/scripts"
        "examples/data"
        "docs/api"
        "docs/tutorials"
        "docs/_static/images"
        "benchmarks/results"
        "deployment/docker"
        "deployment/kubernetes"
        "deployment/cloud"
        "deployment/edge"
        "scripts"
        "data/raw"
        "data/processed"
        "data/models"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_info "Created directory: $dir"
        fi
    done
    
    # Create __init__.py files for Python packages
    python_packages=(
        "wifi_activity_recognition"
        "wifi_activity_recognition/hardware"
        "wifi_activity_recognition/preprocessing"
        "wifi_activity_recognition/features"
        "wifi_activity_recognition/models"
        "wifi_activity_recognition/training"
        "wifi_activity_recognition/datasets"
        "wifi_activity_recognition/inference"
        "wifi_activity_recognition/utils"
        "tests"
        "tests/test_hardware"
        "tests/test_preprocessing"
        "tests/test_models"
        "tests/test_training"
        "tests/test_integration"
        "benchmarks"
    )
    
    for package in "${python_packages[@]}"; do
        init_file="$package/__init__.py"
        if [ ! -f "$init_file" ]; then
            touch "$init_file"
            print_info "Created: $init_file"
        fi
    done
    
    print_success "Project structure created"
    
    print_success "Project structure created"
    
    # Note about configuration files
    print_info "Configuration files can be created as needed during development"
    print_info "See DEVELOPMENT_GUIDE.md for configuration file templates and patterns"
    
    # Note about testing files
    print_info "Test configuration and fixtures can be created as needed"
    print_info "See DEVELOPMENT_GUIDE.md for testing patterns and requirements"
    
    # Create development scripts
    print_header "Creating Development Scripts"
    
    # Create a development helper script
    cat > "scripts/dev.sh" << EOF
#!/bin/bash
# Development helper script

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

case "\$1" in
    "test")
        echo -e "\${GREEN}Running tests...\${NC}"
        python -m pytest tests/ -v
        ;;
    "lint")
        echo -e "\${GREEN}Running code quality checks...\${NC}"
        black --check wifi_activity_recognition/
        flake8 wifi_activity_recognition/
        isort --check-only wifi_activity_recognition/
        mypy wifi_activity_recognition/
        ;;
    "format")
        echo -e "\${GREEN}Formatting code...\${NC}"
        black wifi_activity_recognition/
        isort wifi_activity_recognition/
        ;;
    "docs")
        echo -e "\${GREEN}Building documentation...\${NC}"
        cd docs && make html
        ;;
    "clean")
        echo -e "\${YELLOW}Cleaning build artifacts...\${NC}"
        rm -rf build/ dist/ *.egg-info/
        find . -type d -name __pycache__ -exec rm -rf {} +
        find . -type f -name "*.pyc" -delete
        ;;
    *)
        echo "Usage: \$0 {test|lint|format|docs|clean}"
        exit 1
        ;;
esac
EOF
    
    chmod +x "scripts/dev.sh"
    print_success "Development scripts created"
    
    # Create environment activation script
    cat > "activate.sh" << EOF
#!/bin/bash
# Activate the development environment

echo "Activating WiFi Activity Recognition development environment..."
source $VENV_NAME/bin/activate

echo "Environment activated!"
echo "Python: \$(which python)"
echo "Pip: \$(which pip)"
echo ""
echo "Available commands:"
echo "  wifi-har-* - CLI commands for the package"
echo "  ./scripts/dev.sh test - Run tests"
echo "  ./scripts/dev.sh lint - Check code quality"
echo "  ./scripts/dev.sh format - Format code"
echo ""
echo "To deactivate: deactivate"
EOF
    
    chmod +x "activate.sh"
    print_success "Environment activation script created"
    
    # Final verification
    print_header "Final Verification"
    
    print_info "Verifying installation..."
    
    # Test CLI availability (without running tests that may not exist yet)
    if python -c "import wifi_activity_recognition" 2>/dev/null; then
        print_success "Package import: OK"
    else
        print_warning "Package import not yet available (will be implemented by agents)"
    fi
    
    # Test CLI (may not exist yet)
    if python -c "from wifi_activity_recognition import cli" 2>/dev/null; then
        print_success "CLI module accessible"
    else
        print_info "CLI will be implemented during project development"
    fi
    
    # Check code quality tools
    black --version > /dev/null && print_success "Black formatter available"
    flake8 --version > /dev/null && print_success "Flake8 linter available"
    mypy --version > /dev/null && print_success "MyPy type checker available"
    pytest --version > /dev/null && print_success "Pytest testing framework available"
    
    # Success message
    print_header "Setup Complete!"
    
    echo -e "${GREEN}✓ Development environment successfully set up!${NC}\n"
    
    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "1. Activate environment: ${YELLOW}source activate.sh${NC}"
    echo -e "2. Read the development guide: ${YELLOW}cat DEVELOPMENT_GUIDE.md${NC}"
    echo -e "3. Start implementing components according to ROADMAP.md"
    echo -e "4. Run tests frequently: ${YELLOW}./scripts/dev.sh test${NC}"
    echo -e "5. Check code quality: ${YELLOW}./scripts/dev.sh lint${NC}"
    echo ""
    
    echo -e "${CYAN}Project Structure Created:${NC}"
    echo -e "📁 wifi_activity_recognition/ - Main package code"
    echo -e "📁 tests/ - Test suites with fixtures"
    echo -e "📁 examples/ - Usage examples and notebooks"
    echo -e "📁 docs/ - Documentation"
    echo -e "📁 scripts/ - Development utilities"
    echo -e "📁 configs/ - Configuration files"
    echo ""
    
    echo -e "${CYAN}Development Workflow:${NC}"
    echo -e "• Follow patterns in DEVELOPMENT_GUIDE.md"
    echo -e "• Implement hardware drivers first (Intel 5300, ESP32)"
    echo -e "• Add models (CNN2D, ResNet variants)"
    echo -e "• Build preprocessing pipeline"
    echo -e "• Create training framework"
    echo -e "• Add comprehensive tests"
    echo ""
    
    echo -e "${GREEN}Ready for development! 🚀${NC}"
}

# Handle script interruption
trap 'echo -e "\n${RED}Setup interrupted!${NC}"; exit 1' INT

# Run main setup
main "$@"
