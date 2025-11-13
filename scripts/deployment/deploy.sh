#!/usr/bin/env bash

################################################################################
# Quantum Trader AI Deployment Script
#
# This script manages the deployment of the Quantum Trader AI system,
# including service restart, health checks, and rollback capabilities.
#
# Usage: ./deploy.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help message
#   -v, --version VERSION   Version to deploy (required)
#   -e, --env ENV           Environment (development/staging/production)
#   -s, --service SERVICE   Service to deploy (all/api/worker/scheduler)
#   -r, --rollback          Rollback to previous version
#   -n, --no-health-check   Skip health checks
#   --verbose               Enable verbose output
#
# Examples:
#   ./deploy.sh --version v1.2.0 --env production
#   ./deploy.sh --version v1.2.0 --service api
#   ./deploy.sh --rollback --env production
#
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVICES_DIR="/opt/quantum_trader"
BACKUP_DIR="/opt/quantum_trader/backups"
VERSION=""
ENVIRONMENT="${ENVIRONMENT:-staging}"
SERVICE="${SERVICE:-all}"
ROLLBACK="${ROLLBACK:-0}"
SKIP_HEALTH_CHECK="${SKIP_HEALTH_CHECK:-0}"
VERBOSE="${VERBOSE:-0}"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

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
    head -n 26 "$0" | tail -n +2 | sed 's/^# //'
}

################################################################################
# Backup & Rollback Functions
################################################################################

backup_current_version() {
    log_info "Creating backup of current deployment..."

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="${BACKUP_DIR}/deployment_${timestamp}"

    mkdir -p "${BACKUP_DIR}"

    if cp -r "${SERVICES_DIR}/current" "${backup_path}"; then
        log_success "Backup created at ${backup_path}"
        echo "${backup_path}"
        return 0
    else
        log_error "Failed to create backup"
        return 1
    fi
}

rollback_deployment() {
    log_info "Rolling back to previous version..."

    local previous_backup
    previous_backup=$(find "${BACKUP_DIR}" -maxdepth 1 -name "deployment_*" -type d | sort -r | head -n 1)

    if [[ -z ${previous_backup} ]]; then
        log_error "No previous backup found for rollback"
        return 1
    fi

    log_info "Restoring from backup: ${previous_backup}"

    # Stop services
    if ! systemctl stop quantum-trader &> /dev/null; then
        log_warning "Failed to stop services, continuing with rollback"
    fi

    # Restore files
    if rm -rf "${SERVICES_DIR}/current" && cp -r "${previous_backup}" "${SERVICES_DIR}/current"; then
        log_success "Rollback completed"
        return 0
    else
        log_error "Rollback failed"
        return 1
    fi
}

################################################################################
# Deployment Functions
################################################################################

verify_version_format() {
    local version=$1

    if [[ ! ${version} =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        log_error "Invalid version format: ${version}"
        log_info "Expected format: vX.Y.Z (e.g., v1.2.0)"
        return 1
    fi

    return 0
}

pull_release_artifacts() {
    log_info "Pulling release artifacts for version ${VERSION}..."

    local release_url="https://releases.example.com/quantum-trader/${VERSION}/quantum-trader-${VERSION}.tar.gz"
    local artifact_path="${PROJECT_ROOT}/dist/quantum-trader-${VERSION}.tar.gz"

    mkdir -p "$(dirname "${artifact_path}")"

    if curl -f -L -o "${artifact_path}" "${release_url}"; then
        log_success "Artifacts downloaded"
        return 0
    else
        log_error "Failed to download artifacts"
        return 1
    fi
}

extract_and_prepare_release() {
    log_info "Preparing release ${VERSION}..."

    local artifact_path="${PROJECT_ROOT}/dist/quantum-trader-${VERSION}.tar.gz"
    local extract_dir="${PROJECT_ROOT}/dist/quantum-trader-${VERSION}"

    if tar -xzf "${artifact_path}" -C "$(dirname "${extract_dir}")"; then
        log_success "Release prepared"
        echo "${extract_dir}"
        return 0
    else
        log_error "Failed to extract release"
        return 1
    fi
}

migrate_database() {
    log_info "Running database migrations..."

    if cd "${SERVICES_DIR}/current" && python -m alembic upgrade head; then
        log_success "Database migrations completed"
        return 0
    else
        log_error "Database migrations failed"
        return 1
    fi
}

deploy_service() {
    local service=$1
    log_info "Deploying service: ${service}..."

    case ${service} in
        api)
            systemctl restart quantum-trader-api || return 1
            ;;
        worker)
            systemctl restart quantum-trader-worker || return 1
            ;;
        scheduler)
            systemctl restart quantum-trader-scheduler || return 1
            ;;
        all)
            systemctl restart quantum-trader || return 1
            ;;
        *)
            log_error "Unknown service: ${service}"
            return 1
            ;;
    esac

    log_success "Service ${service} deployed"
    return 0
}

wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=0

    log_info "Waiting for service to be ready..."

    while [[ ${attempt} -lt ${max_attempts} ]]; do
        if systemctl is-active --quiet "${service}"; then
            log_success "Service is ready"
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 2
    done

    log_error "Service did not become ready in time"
    return 1
}

run_health_checks() {
    if [[ ${SKIP_HEALTH_CHECK} -eq 1 ]]; then
        log_warning "Skipping health checks"
        return 0
    fi

    log_info "Running health checks..."

    local health_script="${SCRIPT_DIR}/health_check.sh"

    if [[ ! -f ${health_script} ]]; then
        log_warning "Health check script not found at ${health_script}"
        return 0
    fi

    if bash "${health_script}" --env "${ENVIRONMENT}"; then
        log_success "Health checks passed"
        return 0
    else
        log_error "Health checks failed"
        return 1
    fi
}

cleanup_old_artifacts() {
    log_info "Cleaning up old artifacts..."

    local artifact_dir="${PROJECT_ROOT}/dist"
    local keep_count=5

    if [[ -d ${artifact_dir} ]]; then
        find "${artifact_dir}" -maxdepth 1 -name "quantum-trader-*.tar.gz" -type f | sort -r | tail -n +$((keep_count + 1)) | xargs rm -f
        log_success "Old artifacts cleaned up"
    fi

    return 0
}

################################################################################
# Main Deployment Flow
################################################################################

main() {
    log_info "=== Quantum Trader AI Deployment ==="
    log_info "Environment: ${ENVIRONMENT}"
    log_info ""

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--service)
                SERVICE="$2"
                shift 2
                ;;
            -r|--rollback)
                ROLLBACK=1
                shift
                ;;
            -n|--no-health-check)
                SKIP_HEALTH_CHECK=1
                shift
                ;;
            --verbose)
                VERBOSE=1
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Validate deployment
    if [[ ${ROLLBACK} -eq 0 && -z ${VERSION} ]]; then
        log_error "Version is required for deployment"
        show_help
        exit 1
    fi

    if [[ ${ROLLBACK} -eq 1 ]]; then
        rollback_deployment || exit 1
        run_health_checks || exit 1
        log_success "=== Rollback Complete ==="
        exit 0
    fi

    # Standard deployment flow
    verify_version_format "${VERSION}" || exit 1

    backup_current_version || exit 1
    pull_release_artifacts || exit 1

    local release_dir
    release_dir=$(extract_and_prepare_release) || exit 1

    # Copy to service directory
    log_info "Installing version ${VERSION}..."
    mkdir -p "${SERVICES_DIR}"
    cp -r "${release_dir}" "${SERVICES_DIR}/current"

    migrate_database || exit 1
    deploy_service "${SERVICE}" || exit 1
    wait_for_service "quantum-trader" || exit 1
    run_health_checks || exit 1
    cleanup_old_artifacts || exit 1

    log_info ""
    log_success "=== Deployment Complete ==="
    log_info "Version deployed: ${VERSION}"
    log_info "Environment: ${ENVIRONMENT}"

    exit 0
}

# Run main function
main "$@"
