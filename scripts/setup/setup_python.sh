#!/usr/bin/env bash
# Python 3.12+ Setup Script
# Installs and configures Python 3.12+ with required tools and virtual environment

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
MIN_PYTHON_VERSION="3.12"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}"
}

# Error handler
error_exit() {
    log "ERROR" "$@"
    echo -e "${RED}Error: $@${NC}" >&2
    exit 1
}

# Success message
success() {
    echo -e "${GREEN}✓ $@${NC}"
    log "INFO" "$@"
}

# Warning message
warning() {
    echo -e "${YELLOW}⚠ $@${NC}"
    log "WARN" "$@"
}

# Info message
info() {
    echo -e "${BLUE}ℹ $@${NC}"
    log "INFO" "$@"
}

# Check for root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        warning "Running as root, some operations may require different permissions"
    fi
}

# Detect distribution
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        echo "$ID"
    else
        error_exit "Cannot detect Linux distribution"
    fi
}

# Check Python availability
check_python_available() {
    log "INFO" "Checking for Python ${PYTHON_VERSION}..."

    if command -v "python${PYTHON_VERSION}" &> /dev/null; then
        local python_path
        python_path=$(command -v "python${PYTHON_VERSION}")
        success "Found Python at: $python_path"
        return 0
    fi

    return 1
}

# Install Python 3.12+ from deadsnakes PPA (Ubuntu/Debian)
install_python_from_ppa() {
    log "INFO" "Installing Python ${PYTHON_VERSION} from deadsnakes PPA..."

    if [[ $EUID -ne 0 ]]; then
        error_exit "Installing Python from PPA requires root privileges"
    fi

    local distro
    distro=$(detect_distro)

    case "$distro" in
        ubuntu)
            apt-get update
            apt-get install -y software-properties-common
            add-apt-repository -y "ppa:deadsnakes/ppa"
            apt-get update
            apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev"
            ;;
        debian)
            apt-get update
            apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev"
            ;;
        *)
            error_exit "Unsupported distribution for automatic Python installation: $distro"
            ;;
    esac

    success "Python ${PYTHON_VERSION} installed"
}

# Verify Python version
verify_python_version() {
    local python_cmd="python${PYTHON_VERSION}"

    log "INFO" "Verifying Python version..."

    if ! command -v "$python_cmd" &> /dev/null; then
        error_exit "Python ${PYTHON_VERSION} not found"
    fi

    local python_version
    python_version=$("$python_cmd" --version 2>&1 | awk '{print $2}')

    info "Python version: $python_version"
    success "Python version verified"
}

# Install pip and tools
install_pip_and_tools() {
    local python_cmd="python${PYTHON_VERSION}"

    log "INFO" "Installing pip and Python tools..."

    # Upgrade pip
    "$python_cmd" -m pip install --upgrade pip setuptools wheel

    # Install additional tools
    "$python_cmd" -m pip install --upgrade virtualenv pipenv

    success "Pip and Python tools installed"
}

# Create virtual environment
create_virtual_environment() {
    log "INFO" "Creating virtual environment at: $VENV_DIR"

    local python_cmd="python${PYTHON_VERSION}"

    if [[ -d "$VENV_DIR" ]]; then
        warning "Virtual environment already exists at: $VENV_DIR"
        read -p "Do you want to recreate it? (yes/no): " -r response
        if [[ "$response" =~ ^[Yy][Ee][Ss]$ ]]; then
            rm -rf "$VENV_DIR"
            log "INFO" "Removed existing virtual environment"
        else
            return 0
        fi
    fi

    "$python_cmd" -m venv "$VENV_DIR"

    success "Virtual environment created"
}

# Upgrade virtual environment
upgrade_venv() {
    log "INFO" "Upgrading virtual environment..."

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    pip install --upgrade pip setuptools wheel

    deactivate

    success "Virtual environment upgraded"
}

# Install project dependencies
install_project_dependencies() {
    log "INFO" "Installing project dependencies..."

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    cd "$PROJECT_ROOT"

    # Install base requirements
    if [[ -f "docker/base/requirements-base.txt" ]]; then
        log "INFO" "Installing base requirements..."
        pip install -r docker/base/requirements-base.txt
    fi

    # Install development dependencies
    if [[ -f "requirements-dev.txt" ]]; then
        log "INFO" "Installing development requirements..."
        pip install -r requirements-dev.txt
    fi

    deactivate

    success "Project dependencies installed"
}

# Configure Python environment variables
configure_environment() {
    log "INFO" "Configuring Python environment variables..."

    # Create .env.python file with Python configuration
    cat > "$PROJECT_ROOT/.env.python" <<EOF
# Python Environment Configuration
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=${PROJECT_ROOT}
PYTHONHASHSEED=random
VIRTUAL_ENV=${VENV_DIR}
EOF

    success "Python environment variables configured"
}

# Create activation helper script
create_activation_helper() {
    log "INFO" "Creating activation helper script..."

    cat > "$PROJECT_ROOT/activate_venv.sh" <<EOF
#!/usr/bin/env bash
# Virtual Environment Activation Helper

VENV_DIR="$VENV_DIR"

if [[ ! -d "\$VENV_DIR" ]]; then
    echo "Error: Virtual environment not found at \$VENV_DIR"
    exit 1
fi

# shellcheck disable=SC1091
source "\$VENV_DIR/bin/activate"

# Load Python environment variables if they exist
if [[ -f "${PROJECT_ROOT}/.env.python" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env.python"
fi

echo "Virtual environment activated: \$VENV_DIR"
python --version
EOF

    chmod +x "$PROJECT_ROOT/activate_venv.sh"

    success "Activation helper script created"
}

# Setup IDE support
setup_ide_support() {
    log "INFO" "Setting up IDE support..."

    # Create .vscode/settings.json for VSCode
    mkdir -p "$PROJECT_ROOT/.vscode"

    cat > "$PROJECT_ROOT/.vscode/settings.json" <<EOF
{
    "python.defaultInterpreterPath": "${VENV_DIR}/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-python.python"
    }
}
EOF

    success "IDE support configured (VSCode)"
}

# Generate initialization script
generate_init_script() {
    log "INFO" "Generating initialization script..."

    cat > "$PROJECT_ROOT/setup_python_env.sh" <<'EOF'
#!/usr/bin/env bash
# Python Environment Initialization

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "Initializing Python environment..."

# Activate virtual environment
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Load environment variables
if [[ -f "${SCRIPT_DIR}/.env.python" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env.python"
fi

# Print environment info
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
echo "Virtual Environment: ${VIRTUAL_ENV}"

echo "Python environment initialized successfully!"
EOF

    chmod +x "$PROJECT_ROOT/setup_python_env.sh"

    success "Initialization script generated"
}

# Run final checks
run_final_checks() {
    log "INFO" "Running final checks..."

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    info "Python environment information:"
    echo "  Python Version: $(python --version)"
    echo "  Pip Version: $(pip --version)"
    echo "  Virtual Environment: $VENV_DIR"
    echo "  Python Executable: $(which python)"
    echo "  Pip Executable: $(which pip)"

    # List installed packages
    log "INFO" "Installed packages:"
    pip list | head -20

    deactivate

    success "Python setup completed successfully!"
}

# Main execution
main() {
    log "INFO" "=== Python ${PYTHON_VERSION}+ Setup Started ==="
    log "INFO" "Project root: $PROJECT_ROOT"

    check_root

    local distro
    distro=$(detect_distro)
    log "INFO" "Detected distribution: $distro"

    # Check or install Python
    if ! check_python_available; then
        install_python_from_ppa
    fi

    verify_python_version
    install_pip_and_tools
    create_virtual_environment
    upgrade_venv
    configure_environment
    create_activation_helper
    setup_ide_support
    generate_init_script

    # Optional: install project dependencies
    if [[ -f "$PROJECT_ROOT/docker/base/requirements-base.txt" ]]; then
        install_project_dependencies
    fi

    run_final_checks

    log "INFO" "=== Python ${PYTHON_VERSION}+ Setup Completed ==="

    echo ""
    echo "To activate the virtual environment, run:"
    echo "  source $PROJECT_ROOT/activate_venv.sh"
    echo "or"
    echo "  . $PROJECT_ROOT/.venv/bin/activate"
}

# Run main function
main "$@"
