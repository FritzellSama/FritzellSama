#!/usr/bin/env bash

################################################################################
# Quantum Trader AI Installation Script
#
# This script automates the setup and installation of the Quantum Trader AI
# system, including environment validation, dependency installation, and
# database initialization.
#
# Usage: ./install.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help message
#   -e, --env ENV           Environment (development/staging/production)
#   -p, --python VERSION    Python version (default: 3.9)
#   -v, --verbose           Enable verbose output
#   -d, --dev               Install development dependencies
#
# Examples:
#   ./install.sh
#   ./install.sh --env production --python 3.11
#   ./install.sh --dev --verbose
#
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
ENVIRONMENT="${ENVIRONMENT:-development}"
VERBOSE="${VERBOSE:-0}"
INSTALL_DEV_DEPS="${INSTALL_DEV_DEPS:-0}"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

verbose_log() {
    if [[ ${VERBOSE} -eq 1 ]]; then
        echo -e "${BLUE}[VERBOSE]${NC} $*"
    fi
}

show_help() {
    head -n 25 "$0" | tail -n +2 | sed 's/^# //'
}

################################################################################
# Validation Functions
################################################################################

check_python_installed() {
    log_info "Checking Python installation..."

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        return 1
    fi

    local python_version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    log_success "Found Python ${python_version}"

    return 0
}

check_dependencies() {
    log_info "Checking system dependencies..."

    local missing_deps=()

    # Check for pip
    if ! python3 -m pip --version &> /dev/null; then
        missing_deps+=("pip")
    fi

    # Check for git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    fi

    # Check for required system packages
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies and try again"
        return 1
    fi

    log_success "All required dependencies found"
    return 0
}

################################################################################
# Installation Functions
################################################################################

create_virtual_environment() {
    log_info "Creating virtual environment at ${VENV_DIR}..."

    if [[ -d ${VENV_DIR} ]]; then
        log_warning "Virtual environment already exists, skipping creation"
        return 0
    fi

    if ! python3 -m venv "${VENV_DIR}"; then
        log_error "Failed to create virtual environment"
        return 1
    fi

    log_success "Virtual environment created"
    return 0
}

activate_venv() {
    # shellcheck disable=SC1090,SC1091
    source "${VENV_DIR}/bin/activate"
    verbose_log "Virtual environment activated"
}

upgrade_pip() {
    log_info "Upgrading pip..."

    if ! python3 -m pip install --upgrade pip setuptools wheel &> /dev/null; then
        log_error "Failed to upgrade pip"
        return 1
    fi

    log_success "pip upgraded"
    return 0
}

install_dependencies() {
    log_info "Installing Python dependencies..."

    local requirements_file="${PROJECT_ROOT}/requirements.txt"

    if [[ ! -f ${requirements_file} ]]; then
        log_error "requirements.txt not found at ${requirements_file}"
        return 1
    fi

    if ! pip install -r "${requirements_file}"; then
        log_error "Failed to install dependencies"
        return 1
    fi

    log_success "Python dependencies installed"
    return 0
}

install_dev_dependencies() {
    log_info "Installing development dependencies..."

    local dev_requirements="${PROJECT_ROOT}/requirements-dev.txt"

    if [[ ! -f ${dev_requirements} ]]; then
        log_warning "requirements-dev.txt not found, skipping development dependencies"
        return 0
    fi

    if ! pip install -r "${dev_requirements}"; then
        log_error "Failed to install development dependencies"
        return 1
    fi

    log_success "Development dependencies installed"
    return 0
}

setup_environment_variables() {
    log_info "Setting up environment variables..."

    local env_file="${PROJECT_ROOT}/.env"
    local env_example="${PROJECT_ROOT}/.env.example"

    if [[ -f ${env_file} ]]; then
        log_warning ".env file already exists, skipping"
        return 0
    fi

    if [[ -f ${env_example} ]]; then
        cp "${env_example}" "${env_file}"
        log_success ".env file created from template"
        log_info "Please configure .env with your settings"
        return 0
    fi

    log_warning ".env.example not found, skipping .env setup"
    return 0
}

initialize_database() {
    log_info "Initializing database..."

    if [[ ! -f ${PROJECT_ROOT}/src/quantum_trader/scripts/init_db.py ]]; then
        log_warning "Database initialization script not found, skipping"
        return 0
    fi

    if python -m quantum_trader.scripts.init_db; then
        log_success "Database initialized"
        return 0
    else
        log_warning "Database initialization encountered issues, review the output above"
        return 0
    fi
}

run_tests() {
    log_info "Running test suite..."

    if ! command -v pytest &> /dev/null; then
        log_warning "pytest not found, skipping tests"
        return 0
    fi

    if pytest "${PROJECT_ROOT}/tests" -v --tb=short; then
        log_success "All tests passed"
        return 0
    else
        log_warning "Some tests failed, review the output above"
        return 0
    fi
}

################################################################################
# Main Installation Flow
################################################################################

main() {
    log_info "=== Quantum Trader AI Installation ==="
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Python version: ${PYTHON_VERSION}"
    log_info ""

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -p|--python)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -d|--dev)
                INSTALL_DEV_DEPS=1
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Execute installation steps
    check_python_installed || exit 1
    check_dependencies || exit 1
    create_virtual_environment || exit 1
    activate_venv

    upgrade_pip || exit 1
    install_dependencies || exit 1

    if [[ ${INSTALL_DEV_DEPS} -eq 1 ]]; then
        install_dev_dependencies || exit 1
    fi

    setup_environment_variables || exit 1
    initialize_database || exit 1

    if [[ ${INSTALL_DEV_DEPS} -eq 1 ]]; then
        run_tests || exit 1
    fi

    log_info ""
    log_success "=== Installation Complete ==="
    log_info "Next steps:"
    log_info "  1. Configure your environment variables in .env"
    log_info "  2. Run: source ${VENV_DIR}/bin/activate"
    log_info "  3. Start the application: quantum-trader start"

    exit 0
}

# Run main function
main "$@"
