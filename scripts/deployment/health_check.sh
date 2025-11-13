#!/usr/bin/env bash

################################################################################
# Quantum Trader AI Health Check Script
#
# This script performs comprehensive health checks on the Quantum Trader AI
# deployment, including service availability, database connectivity,
# dependency health, and performance metrics.
#
# Usage: ./health_check.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help message
#   -e, --env ENV           Environment (development/staging/production)
#   -c, --checks CHECKS     Specific checks to run (comma-separated)
#   -w, --warning PERCENT   Warning threshold for metrics (default: 80)
#   -c, --critical PERCENT  Critical threshold for metrics (default: 95)
#   --verbose               Enable verbose output
#
# Available Checks:
#   - api               API service availability
#   - database          Database connectivity
#   - redis             Redis cache connectivity
#   - dependencies      Python dependency checks
#   - disk-space        Available disk space
#   - memory            Memory usage
#   - cpu               CPU usage
#   - network           Network connectivity
#
# Examples:
#   ./health_check.sh --env production
#   ./health_check.sh --env staging --checks api,database,redis
#
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENVIRONMENT="${ENVIRONMENT:-development}"
ALL_CHECKS=("api" "database" "redis" "dependencies" "disk-space" "memory" "cpu" "network")
SELECTED_CHECKS=()
WARNING_THRESHOLD="${WARNING_THRESHOLD:-80}"
CRITICAL_THRESHOLD="${CRITICAL_THRESHOLD:-95}"
VERBOSE="${VERBOSE:-0}"

# Check results
declare -A CHECK_RESULTS=()
OVERALL_STATUS="OK"

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
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

verbose_log() {
    if [[ ${VERBOSE} -eq 1 ]]; then
        echo -e "${BLUE}[VERBOSE]${NC} $*"
    fi
}

show_help() {
    head -n 35 "$0" | tail -n +2 | sed 's/^# //'
}

update_status() {
    local check=$1
    local status=$2

    CHECK_RESULTS["${check}"]="${status}"

    if [[ ${status} == "FAILED" ]] && [[ ${OVERALL_STATUS} != "FAILED" ]]; then
        OVERALL_STATUS="FAILED"
    elif [[ ${status} == "WARNING" ]] && [[ ${OVERALL_STATUS} == "OK" ]]; then
        OVERALL_STATUS="WARNING"
    fi
}

################################################################################
# Service Health Checks
################################################################################

check_api_health() {
    log_info "Checking API service..."

    local api_host="${API_HOST:-localhost}"
    local api_port="${API_PORT:-8000}"
    local max_attempts=3
    local attempt=0

    while [[ ${attempt} -lt ${max_attempts} ]]; do
        if curl -sf "http://${api_host}:${api_port}/health" > /dev/null 2>&1; then
            log_success "API service is healthy"
            update_status "api" "OK"
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 1
    done

    log_error "API service is not responding"
    update_status "api" "FAILED"
    return 1
}

check_database_health() {
    log_info "Checking database connectivity..."

    local db_host="${DATABASE_HOST:-localhost}"
    local db_port="${DATABASE_PORT:-5432}"
    local db_name="${DATABASE_NAME:-quantum_trader}"
    local db_user="${DATABASE_USER:-postgres}"

    if PGPASSWORD="${DATABASE_PASSWORD:-}" psql -h "${db_host}" -p "${db_port}" \
        -U "${db_user}" -d "${db_name}" -c "SELECT 1" > /dev/null 2>&1; then
        log_success "Database is healthy"
        update_status "database" "OK"
        return 0
    else
        log_error "Database connection failed"
        update_status "database" "FAILED"
        return 1
    fi
}

check_redis_health() {
    log_info "Checking Redis cache..."

    local redis_host="${REDIS_HOST:-localhost}"
    local redis_port="${REDIS_PORT:-6379}"

    if redis-cli -h "${redis_host}" -p "${redis_port}" ping > /dev/null 2>&1; then
        log_success "Redis is healthy"
        update_status "redis" "OK"
        return 0
    else
        log_warning "Redis is not available (optional)"
        update_status "redis" "WARNING"
        return 0
    fi
}

check_dependencies() {
    log_info "Checking Python dependencies..."

    # Check if in virtual environment
    if [[ -z ${VIRTUAL_ENV:-} ]]; then
        log_warning "Not running in a virtual environment"
    fi

    # Verify critical imports
    if python3 -c "import quantum_trader; import requests; import sqlalchemy; import redis" 2>/dev/null; then
        log_success "All critical dependencies are available"
        update_status "dependencies" "OK"
        return 0
    else
        log_error "Missing required dependencies"
        update_status "dependencies" "FAILED"
        return 1
    fi
}

################################################################################
# System Health Checks
################################################################################

check_disk_space() {
    log_info "Checking disk space..."

    local disk_usage
    disk_usage=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')

    if [[ ${disk_usage} -ge ${CRITICAL_THRESHOLD} ]]; then
        log_error "Critical disk space: ${disk_usage}% used"
        update_status "disk-space" "FAILED"
        return 1
    elif [[ ${disk_usage} -ge ${WARNING_THRESHOLD} ]]; then
        log_warning "High disk usage: ${disk_usage}%"
        update_status "disk-space" "WARNING"
        return 0
    else
        log_success "Disk space is healthy: ${disk_usage}% used"
        update_status "disk-space" "OK"
        return 0
    fi
}

check_memory_usage() {
    log_info "Checking memory usage..."

    local memory_usage
    memory_usage=$(free | grep Mem | awk '{printf "%.0f", ($3/$2) * 100}')

    if [[ ${memory_usage} -ge ${CRITICAL_THRESHOLD} ]]; then
        log_error "Critical memory usage: ${memory_usage}%"
        update_status "memory" "FAILED"
        return 1
    elif [[ ${memory_usage} -ge ${WARNING_THRESHOLD} ]]; then
        log_warning "High memory usage: ${memory_usage}%"
        update_status "memory" "WARNING"
        return 0
    else
        log_success "Memory usage is healthy: ${memory_usage}%"
        update_status "memory" "OK"
        return 0
    fi
}

check_cpu_usage() {
    log_info "Checking CPU usage..."

    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.0f", 100 - $1}')

    if [[ ${cpu_usage} -ge ${CRITICAL_THRESHOLD} ]]; then
        log_error "Critical CPU usage: ${cpu_usage}%"
        update_status "cpu" "FAILED"
        return 1
    elif [[ ${cpu_usage} -ge ${WARNING_THRESHOLD} ]]; then
        log_warning "High CPU usage: ${cpu_usage}%"
        update_status "cpu" "WARNING"
        return 0
    else
        log_success "CPU usage is healthy: ${cpu_usage}%"
        update_status "cpu" "OK"
        return 0
    fi
}

check_network_connectivity() {
    log_info "Checking network connectivity..."

    # Test connection to common services
    local tests=(
        "8.8.8.8:53"        # Google DNS
        "1.1.1.1:53"         # Cloudflare DNS
    )

    for test in "${tests[@]}"; do
        if timeout 2 bash -c "cat < /dev/null > /dev/tcp/${test%:*}/${test##*:}" 2>/dev/null; then
            log_success "Network connectivity is healthy"
            update_status "network" "OK"
            return 0
        fi
    done

    log_warning "Network connectivity may be limited"
    update_status "network" "WARNING"
    return 0
}

################################################################################
# Report Generation
################################################################################

print_report() {
    log_info ""
    log_info "=== Health Check Report ==="
    log_info "Environment: ${ENVIRONMENT}"
    log_info ""

    # Print individual check results
    for check in "${ALL_CHECKS[@]}"; do
        if [[ -v CHECK_RESULTS["${check}"] ]]; then
            local status="${CHECK_RESULTS[${check}]}"

            case ${status} in
                OK)
                    log_success "${check}"
                    ;;
                WARNING)
                    log_warning "${check}"
                    ;;
                FAILED)
                    log_error "${check}"
                    ;;
            esac
        fi
    done

    log_info ""

    # Print overall status
    case ${OVERALL_STATUS} in
        OK)
            log_success "Overall Status: HEALTHY"
            echo ""
            return 0
            ;;
        WARNING)
            log_warning "Overall Status: DEGRADED (some warnings)"
            echo ""
            return 0
            ;;
        FAILED)
            log_error "Overall Status: UNHEALTHY (critical issues)"
            echo ""
            return 1
            ;;
    esac
}

################################################################################
# Main Health Check Flow
################################################################################

main() {
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
            -c|--checks)
                IFS=',' read -ra SELECTED_CHECKS <<< "$2"
                shift 2
                ;;
            -w|--warning)
                WARNING_THRESHOLD="$2"
                shift 2
                ;;
            -t|--critical)
                CRITICAL_THRESHOLD="$2"
                shift 2
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

    log_info "=== Quantum Trader AI Health Check ==="
    log_info "Environment: ${ENVIRONMENT}"
    log_info ""

    # Determine which checks to run
    local checks_to_run=()
    if [[ ${#SELECTED_CHECKS[@]} -eq 0 ]]; then
        checks_to_run=("${ALL_CHECKS[@]}")
    else
        checks_to_run=("${SELECTED_CHECKS[@]}")
    fi

    # Run selected health checks
    for check in "${checks_to_run[@]}"; do
        case ${check} in
            api)
                check_api_health || true
                ;;
            database)
                check_database_health || true
                ;;
            redis)
                check_redis_health || true
                ;;
            dependencies)
                check_dependencies || true
                ;;
            disk-space)
                check_disk_space || true
                ;;
            memory)
                check_memory_usage || true
                ;;
            cpu)
                check_cpu_usage || true
                ;;
            network)
                check_network_connectivity || true
                ;;
            *)
                log_warning "Unknown check: ${check}"
                ;;
        esac
    done

    # Print report and exit with appropriate status
    print_report
}

# Run main function
main "$@"
