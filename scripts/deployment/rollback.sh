#!/usr/bin/env bash
# Deployment Rollback Script
# Rolls back deployment to a previous version

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/rollback_$(date +%Y%m%d_%H%M%S).log"
DEPLOYMENT_DIR="${DEPLOYMENT_DIR:-${PROJECT_ROOT}/deployments}"
VERSION_TO_ROLLBACK="${1:-}"

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

# Get current version
get_current_version() {
    local version_file="${PROJECT_ROOT}/.version"

    if [[ ! -f "$version_file" ]]; then
        error_exit "Version file not found: $version_file"
    fi

    cat "$version_file"
}

# List available versions
list_available_versions() {
    log "INFO" "Listing available deployment versions..."

    if [[ ! -d "$DEPLOYMENT_DIR" ]]; then
        error_exit "Deployment directory not found: $DEPLOYMENT_DIR"
    fi

    echo -e "${BLUE}Available deployment versions:${NC}"
    local current_version
    current_version=$(get_current_version)

    ls -1t "$DEPLOYMENT_DIR" | head -20 | nl | while read -r num version_dir; do
        if [[ "$version_dir" == "$current_version" ]]; then
            echo -e "  ${GREEN}${num}. ${version_dir} [CURRENT]${NC}"
        else
            echo "  ${num}. ${version_dir}"
        fi
    done
}

# Validate version
validate_version() {
    local version="$1"

    if [[ ! -d "${DEPLOYMENT_DIR}/${version}" ]]; then
        error_exit "Version directory not found: ${DEPLOYMENT_DIR}/${version}"
    fi

    if [[ ! -f "${DEPLOYMENT_DIR}/${version}/docker-compose.yml" ]]; then
        error_exit "No docker-compose.yml found for version: $version"
    fi

    log "INFO" "Version $version validated successfully"
}

# Create rollback snapshot
create_rollback_snapshot() {
    local current_version="$1"

    log "INFO" "Creating rollback snapshot of current version..."

    local snapshot_dir="${DEPLOYMENT_DIR}/.snapshots"
    mkdir -p "$snapshot_dir"

    local snapshot_file="${snapshot_dir}/snapshot_${current_version}_$(date +%Y%m%d_%H%M%S).tar.gz"

    cd "$PROJECT_ROOT"
    tar -czf "$snapshot_file" \
        docker-compose.yml \
        .env \
        logs/ \
        2>/dev/null || warning "Could not create complete snapshot"

    log "INFO" "Snapshot created: $snapshot_file"
    echo "$snapshot_file"
}

# Perform health check
health_check() {
    log "INFO" "Performing health check..."

    cd "$PROJECT_ROOT"

    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose ps --services --filter "status=running" | grep -q . && \
           curl -sf http://localhost:8000/health &>/dev/null; then
            success "Health check passed"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done

    warning "Health check failed or timed out"
    return 1
}

# Rollback version
rollback_version() {
    local target_version="$1"

    log "INFO" "Rolling back to version: $target_version"

    cd "$PROJECT_ROOT"

    # Stop current services
    log "INFO" "Stopping current services..."
    docker-compose down --timeout=30 2>&1 | tee -a "$LOG_FILE"

    success "Services stopped"

    # Copy target version files
    log "INFO" "Copying target version files..."
    cp "${DEPLOYMENT_DIR}/${target_version}/docker-compose.yml" . || error_exit "Could not copy docker-compose.yml"

    if [[ -f "${DEPLOYMENT_DIR}/${target_version}/.env" ]]; then
        cp "${DEPLOYMENT_DIR}/${target_version}/.env" . || warning "Could not copy .env file"
    fi

    # Update version file
    echo "$target_version" > .version

    success "Version files updated"

    # Pull required images
    log "INFO" "Pulling Docker images..."
    docker-compose pull 2>&1 | tee -a "$LOG_FILE"

    # Start services with target version
    log "INFO" "Starting services with target version..."
    docker-compose up -d 2>&1 | tee -a "$LOG_FILE"

    success "Services started with target version: $target_version"
}

# Verify rollback
verify_rollback() {
    local target_version="$1"

    log "INFO" "Verifying rollback..."

    if health_check; then
        local current_version
        current_version=$(get_current_version)
        if [[ "$current_version" == "$target_version" ]]; then
            success "Rollback verified successfully"
            return 0
        fi
    fi

    error_exit "Rollback verification failed"
}

# Confirm action
confirm_rollback() {
    local current_version="$1"
    local target_version="$2"

    echo ""
    echo -e "${YELLOW}=== Rollback Confirmation ===${NC}"
    echo "Current version: $current_version"
    echo "Target version:  $target_version"
    echo ""

    read -p "Do you want to proceed with the rollback? (yes/no): " -r response
    if [[ ! "$response" =~ ^[Yy][Ee][Ss]$ ]]; then
        log "INFO" "Rollback cancelled by user"
        error_exit "Rollback cancelled"
    fi
}

# Main execution
main() {
    log "INFO" "=== Deployment Rollback Started ==="
    log "INFO" "Project root: $PROJECT_ROOT"

    local current_version
    current_version=$(get_current_version)

    log "INFO" "Current deployed version: $current_version"

    # If no version specified, show list and prompt
    if [[ -z "$VERSION_TO_ROLLBACK" ]]; then
        list_available_versions
        echo ""
        read -p "Enter version to rollback to: " -r VERSION_TO_ROLLBACK
        if [[ -z "$VERSION_TO_ROLLBACK" ]]; then
            error_exit "No version specified"
        fi
    fi

    # Validate target version
    validate_version "$VERSION_TO_ROLLBACK"

    # Confirm action
    confirm_rollback "$current_version" "$VERSION_TO_ROLLBACK"

    # Create snapshot of current version
    create_rollback_snapshot "$current_version"

    # Perform rollback
    rollback_version "$VERSION_TO_ROLLBACK"

    # Verify rollback
    verify_rollback "$VERSION_TO_ROLLBACK"

    log "INFO" "=== Deployment Rollback Completed Successfully ==="
    success "Rollback to version $VERSION_TO_ROLLBACK completed!"
    info "Check logs at $LOG_FILE for details"
}

# Run main function
main "$@"
