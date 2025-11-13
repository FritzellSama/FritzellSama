#!/usr/bin/env bash
# Dependencies Update Script
# Updates Python package dependencies and checks for security vulnerabilities

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
LOG_FILE="${PROJECT_ROOT}/logs/dependencies_$(date +%Y%m%d_%H%M%S).log"
CHECK_SECURITY="${CHECK_SECURITY:-true}"
BACKUP_REQUIREMENTS="${BACKUP_REQUIREMENTS:-true}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure logs directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    local level="$1"
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "$LOG_FILE"
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

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."

    if [[ ! -d "$VENV_DIR" ]]; then
        error_exit "Virtual environment not found at: $VENV_DIR"
    fi

    if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
        error_exit "Virtual environment activation script not found"
    fi

    success "Prerequisites check passed"
}

# Activate virtual environment
activate_venv() {
    log "INFO" "Activating virtual environment..."

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    success "Virtual environment activated"
}

# Get list of requirements files
get_requirements_files() {
    log "INFO" "Finding requirements files..."

    local files=()

    # Add main requirements files
    if [[ -f "$PROJECT_ROOT/docker/base/requirements-base.txt" ]]; then
        files+=("$PROJECT_ROOT/docker/base/requirements-base.txt")
    fi

    if [[ -f "$PROJECT_ROOT/docker/bot/requirements-bot.txt" ]]; then
        files+=("$PROJECT_ROOT/docker/bot/requirements-bot.txt")
    fi

    if [[ -f "$PROJECT_ROOT/docker/dashboard/requirements-dashboard.txt" ]]; then
        files+=("$PROJECT_ROOT/docker/dashboard/requirements-dashboard.txt")
    fi

    if [[ -f "$PROJECT_ROOT/docker/ml/requirements-ml.txt" ]]; then
        files+=("$PROJECT_ROOT/docker/ml/requirements-ml.txt")
    fi

    if [[ -f "$PROJECT_ROOT/docker/monitoring/requirements-monitoring.txt" ]]; then
        files+=("$PROJECT_ROOT/docker/monitoring/requirements-monitoring.txt")
    fi

    if [[ -f "$PROJECT_ROOT/requirements-dev.txt" ]]; then
        files+=("$PROJECT_ROOT/requirements-dev.txt")
    fi

    if [[ ${#files[@]} -eq 0 ]]; then
        warning "No requirements files found"
        return 1
    fi

    printf '%s\n' "${files[@]}" | tee -a "$LOG_FILE"
    return 0
}

# Backup requirements file
backup_requirements_file() {
    local requirements_file="$1"

    if [[ ! -f "$requirements_file" ]]; then
        return 0
    fi

    if [[ "$BACKUP_REQUIREMENTS" != "true" ]]; then
        return 0
    fi

    local backup_dir="${PROJECT_ROOT}/.backups/requirements"
    mkdir -p "$backup_dir"

    local backup_file="${backup_dir}/$(basename "$requirements_file").$(date +%Y%m%d_%H%M%S)"

    cp "$requirements_file" "$backup_file"

    log "INFO" "Backup created: $backup_file"
}

# Update single requirements file
update_requirements_file() {
    local requirements_file="$1"

    if [[ ! -f "$requirements_file" ]]; then
        log "WARN" "Requirements file not found: $requirements_file"
        return 1
    fi

    log "INFO" "Updating: $(basename "$requirements_file")"

    # Backup the file
    backup_requirements_file "$requirements_file"

    # Install current requirements
    log "INFO" "Installing packages from: $(basename "$requirements_file")"
    pip install -r "$requirements_file" --upgrade 2>&1 | tee -a "$LOG_FILE"

    success "Updated: $(basename "$requirements_file")"
}

# Freeze current environment
freeze_environment() {
    log "INFO" "Freezing current environment..."

    local freeze_file="${PROJECT_ROOT}/requirements-frozen.txt"

    pip freeze > "$freeze_file"

    log "INFO" "Environment frozen to: $freeze_file"
}

# Check for security vulnerabilities
check_security_vulnerabilities() {
    if [[ "$CHECK_SECURITY" != "true" ]]; then
        return 0
    fi

    log "INFO" "Checking for security vulnerabilities..."

    # Install safety if not available
    if ! pip list | grep -q safety; then
        log "INFO" "Installing safety for vulnerability checks..."
        pip install safety -q
    fi

    # Run safety check
    if safety check --json 2>&1 | tee -a "$LOG_FILE"; then
        success "No security vulnerabilities found"
    else
        warning "Security vulnerabilities detected - review logs"
    fi
}

# Check for outdated packages
check_outdated_packages() {
    log "INFO" "Checking for outdated packages..."

    echo "" | tee -a "$LOG_FILE"
    echo "=== Outdated Packages ===" | tee -a "$LOG_FILE"

    local outdated_count=0
    while IFS= read -r line; do
        if [[ ! -z "$line" ]]; then
            echo "  $line" | tee -a "$LOG_FILE"
            outdated_count=$((outdated_count + 1))
        fi
    done < <(pip list --outdated --format=json 2>/dev/null | jq -r '.[] | "\(.name) (\(.version} -> \(.latest_version})"' 2>/dev/null || pip list --outdated)

    if [[ $outdated_count -gt 0 ]]; then
        info "$outdated_count packages can be upgraded"
    else
        success "All packages are up to date"
    fi
}

# Generate dependency report
generate_dependency_report() {
    log "INFO" "Generating dependency report..."

    local report_file="${PROJECT_ROOT}/dependency_report_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$report_file" <<EOF
=== Dependency Update Report ===
Generated: $(date '+%Y-%m-%d %H:%M:%S')

Python Version: $(python --version)
Pip Version: $(pip --version)

=== Installed Packages ===
EOF

    pip list >> "$report_file"

    log "INFO" "Report generated: $report_file"
}

# Check for conflicts
check_dependency_conflicts() {
    log "INFO" "Checking for dependency conflicts..."

    # Install pipdeptree if not available
    if ! pip list | grep -q pipdeptree; then
        log "INFO" "Installing pipdeptree for conflict detection..."
        pip install pipdeptree -q
    fi

    # Check for conflicts
    if pipdeptree --warn fail 2>&1 | tee -a "$LOG_FILE"; then
        success "No dependency conflicts found"
    else
        warning "Dependency conflicts detected - review logs"
    fi
}

# Validate installation
validate_installation() {
    log "INFO" "Validating installation..."

    # Try importing key modules
    local test_imports=("flask" "pandas" "numpy" "requests")

    for module in "${test_imports[@]}"; do
        if python -c "import $module" 2>/dev/null; then
            log "INFO" "✓ $module is importable"
        else
            warning "⚠ $module is not importable"
        fi
    done

    success "Validation completed"
}

# Clean up pip cache
cleanup_pip_cache() {
    log "INFO" "Cleaning up pip cache..."

    pip cache purge 2>/dev/null || pip install --upgrade pip -q

    success "Pip cache cleaned"
}

# Generate update summary
generate_update_summary() {
    log "INFO" "Generating update summary..."

    echo "" | tee -a "$LOG_FILE"
    echo "=== Update Summary ===" | tee -a "$LOG_FILE"
    echo "Total installed packages: $(pip list | wc -l)" | tee -a "$LOG_FILE"
    echo "Python version: $(python --version)" | tee -a "$LOG_FILE"
    echo "Pip version: $(pip --version)" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
}

# Main execution
main() {
    log "INFO" "=== Dependencies Update Started ==="
    log "INFO" "Project root: $PROJECT_ROOT"
    log "INFO" "Virtual environment: $VENV_DIR"

    check_prerequisites
    activate_venv

    # Get requirements files
    local requirements_files
    if ! requirements_files=$(get_requirements_files); then
        warning "No requirements files found, proceeding with pip update"
    else
        # Update each requirements file
        while IFS= read -r requirements_file; do
            update_requirements_file "$requirements_file"
        done <<< "$requirements_files"
    fi

    # Freeze environment
    freeze_environment

    # Security checks
    check_security_vulnerabilities

    # Dependency analysis
    check_outdated_packages
    check_dependency_conflicts

    # Validation
    validate_installation

    # Cleanup
    cleanup_pip_cache

    # Generate reports
    generate_dependency_report
    generate_update_summary

    log "INFO" "=== Dependencies Update Completed Successfully ==="
    success "Dependencies updated! Check logs at $LOG_FILE"
}

# Run main function
main "$@"
