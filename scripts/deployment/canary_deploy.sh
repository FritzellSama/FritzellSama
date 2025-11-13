#!/usr/bin/env bash

################################################################################
# Quantum Trader AI Canary Deployment Script
#
# This script implements canary deployment strategy, gradually rolling out
# a new version to a small percentage of traffic/users before full deployment.
#
# Usage: ./canary_deploy.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help message
#   -v, --version VERSION   Version to deploy (required)
#   -c, --canary PERCENT    Canary traffic percentage (1-50, default: 10)
#   -i, --interval SEC      Monitoring interval in seconds (default: 60)
#   -d, --duration MIN      Total duration in minutes (default: 10)
#   -t, --threshold PERCENT Error rate threshold for rollback (default: 5)
#   -e, --env ENV           Environment (staging/production, default: staging)
#   --verbose               Enable verbose output
#
# Examples:
#   ./canary_deploy.sh --version v1.2.0 --canary 10 --duration 15
#   ./canary_deploy.sh --version v1.2.0 --canary 25 --threshold 3 --env production
#
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERSION=""
CANARY_PERCENT="${CANARY_PERCENT:-10}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
MONITOR_DURATION="${MONITOR_DURATION:-10}"
ERROR_THRESHOLD="${ERROR_THRESHOLD:-5}"
ENVIRONMENT="${ENVIRONMENT:-staging}"
VERBOSE="${VERBOSE:-0}"

# Tracking variables
CANARY_REPLICAS=0
STABLE_REPLICAS=0
START_TIME=0
METRICS_DATA=()

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

elapsed_time() {
    local current_time
    current_time=$(date +%s)
    echo $((current_time - START_TIME))
}

format_duration() {
    local seconds=$1
    local minutes=$((seconds / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d" "${minutes}" "${secs}"
}

################################################################################
# Kubernetes Functions
################################################################################

get_current_replicas() {
    local deployment=$1
    kubectl get deployment "${deployment}" -n quantum-trader \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0"
}

get_ready_replicas() {
    local deployment=$1
    kubectl get deployment "${deployment}" -n quantum-trader \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0"
}

scale_deployment() {
    local deployment=$1
    local replicas=$2

    log_info "Scaling ${deployment} to ${replicas} replicas..."

    if kubectl scale deployment "${deployment}" --replicas="${replicas}" -n quantum-trader; then
        log_success "Scaled to ${replicas} replicas"
        return 0
    else
        log_error "Failed to scale deployment"
        return 1
    fi
}

wait_for_ready() {
    local deployment=$1
    local expected_replicas=$2
    local max_attempts=30
    local attempt=0

    log_info "Waiting for ${deployment} to be ready..."

    while [[ ${attempt} -lt ${max_attempts} ]]; do
        local ready_replicas
        ready_replicas=$(get_ready_replicas "${deployment}")

        if [[ ${ready_replicas} -ge ${expected_replicas} ]]; then
            log_success "${deployment} is ready (${ready_replicas} replicas)"
            return 0
        fi

        verbose_log "Ready replicas: ${ready_replicas}/${expected_replicas}"
        attempt=$((attempt + 1))
        sleep 2
    done

    log_error "Deployment did not become ready in time"
    return 1
}

################################################################################
# Metrics & Monitoring Functions
################################################################################

get_error_rate() {
    local service=$1
    local minutes=1

    # Query Prometheus for error rate
    local query="rate(quantum_trader_http_errors_total[${minutes}m])"

    verbose_log "Querying error rate for ${service}..."

    # Simulated query - replace with actual Prometheus query
    local error_rate=$(curl -s "http://localhost:9090/api/v1/query" \
        --data-urlencode "query=${query}" 2>/dev/null | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    echo "${error_rate}"
}

get_latency() {
    local service=$1

    # Query Prometheus for p99 latency
    local query="histogram_quantile(0.99, rate(quantum_trader_http_duration_seconds_bucket[1m]))"

    verbose_log "Querying latency for ${service}..."

    # Simulated query - replace with actual Prometheus query
    local latency=$(curl -s "http://localhost:9090/api/v1/query" \
        --data-urlencode "query=${query}" 2>/dev/null | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    echo "${latency}"
}

check_health() {
    local deployment=$1

    verbose_log "Running health check for ${deployment}..."

    if kubectl exec -n quantum-trader \
        "deployment/${deployment}" -- curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

collect_metrics() {
    local error_rate
    error_rate=$(get_error_rate "quantum-trader-canary")

    local timestamp
    timestamp=$(date +%s)

    METRICS_DATA+=("${timestamp}|${error_rate}")

    log_info "Metrics collected - Error Rate: ${error_rate}%"
}

analyze_metrics() {
    log_info "Analyzing canary metrics..."

    if [[ ${#METRICS_DATA[@]} -lt 3 ]]; then
        log_warning "Insufficient data for analysis"
        return 0
    fi

    # Calculate average error rate from recent measurements
    local error_rates=()
    for data in "${METRICS_DATA[@]}"; do
        error_rates+=("$(echo "${data}" | cut -d'|' -f2)")
    done

    local total_error=0
    for rate in "${error_rates[@]}"; do
        total_error=$(echo "${total_error} + ${rate}" | bc)
    done

    local avg_error
    avg_error=$(echo "scale=2; ${total_error} / ${#error_rates[@]}" | bc)

    log_info "Average error rate: ${avg_error}%"

    if (( $(echo "${avg_error} > ${ERROR_THRESHOLD}" | bc -l) )); then
        log_error "Error rate exceeded threshold (${avg_error}% > ${ERROR_THRESHOLD}%)"
        return 1
    fi

    return 0
}

################################################################################
# Canary Deployment Functions
################################################################################

verify_version_exists() {
    log_info "Verifying version ${VERSION} exists..."

    if docker pull "your-registry/quantum-trader:${VERSION}" > /dev/null 2>&1; then
        log_success "Version ${VERSION} found"
        return 0
    else
        log_error "Version ${VERSION} not found in registry"
        return 1
    fi
}

create_canary_deployment() {
    log_info "Creating canary deployment for version ${VERSION}..."

    # Calculate replica counts
    local total_replicas
    total_replicas=$(get_current_replicas "quantum-trader-stable")

    STABLE_REPLICAS=$((total_replicas * (100 - CANARY_PERCENT) / 100))
    CANARY_REPLICAS=$((total_replicas * CANARY_PERCENT / 100))

    # Ensure at least 1 canary replica
    if [[ ${CANARY_REPLICAS} -lt 1 ]]; then
        CANARY_REPLICAS=1
    fi

    log_info "Deployment plan: ${CANARY_REPLICAS} canary, ${STABLE_REPLICAS} stable"

    # Scale stable deployment down
    scale_deployment "quantum-trader-stable" "${STABLE_REPLICAS}" || return 1

    # Create/update canary deployment
    if kubectl set image deployment/quantum-trader-canary \
        quantum-trader="your-registry/quantum-trader:${VERSION}" \
        -n quantum-trader --record; then

        scale_deployment "quantum-trader-canary" "${CANARY_REPLICAS}" || return 1
        wait_for_ready "quantum-trader-canary" "${CANARY_REPLICAS}" || return 1
        return 0
    else
        log_error "Failed to update canary deployment"
        return 1
    fi
}

monitor_canary() {
    log_info "Monitoring canary deployment..."
    log_info "Duration: ${MONITOR_DURATION} minutes, Interval: ${MONITOR_INTERVAL}s"

    START_TIME=$(date +%s)
    local max_iterations=$((MONITOR_DURATION * 60 / MONITOR_INTERVAL))
    local iteration=0

    while [[ ${iteration} -lt ${max_iterations} ]]; do
        local elapsed
        elapsed=$(elapsed_time)
        local formatted
        formatted=$(format_duration "${elapsed}")

        log_info "[${formatted}] Checking canary health..."

        # Collect and analyze metrics
        collect_metrics

        if ! analyze_metrics; then
            log_error "Canary metrics indicate issues, initiating rollback"
            return 1
        fi

        # Check pod health
        if ! check_health "quantum-trader-canary"; then
            log_error "Canary health check failed"
            return 1
        fi

        iteration=$((iteration + 1))

        if [[ ${iteration} -lt ${max_iterations} ]]; then
            verbose_log "Sleeping for ${MONITOR_INTERVAL} seconds..."
            sleep "${MONITOR_INTERVAL}"
        fi
    done

    log_success "Canary monitoring completed successfully"
    return 0
}

promote_canary() {
    log_info "Promoting canary to stable..."

    local total_replicas
    total_replicas=$((CANARY_REPLICAS + STABLE_REPLICAS))

    # Update stable deployment with new version
    if kubectl set image deployment/quantum-trader-stable \
        quantum-trader="your-registry/quantum-trader:${VERSION}" \
        -n quantum-trader --record; then

        scale_deployment "quantum-trader-stable" "${total_replicas}" || return 1
        wait_for_ready "quantum-trader-stable" "${total_replicas}" || return 1

        # Remove canary deployment
        scale_deployment "quantum-trader-canary" 0 || log_warning "Failed to scale down canary"

        log_success "Canary promoted to stable"
        return 0
    else
        log_error "Failed to promote canary"
        return 1
    fi
}

rollback_canary() {
    log_warning "Rolling back canary deployment..."

    local original_replicas
    original_replicas=$((CANARY_REPLICAS + STABLE_REPLICAS))

    # Scale down canary
    scale_deployment "quantum-trader-canary" 0 || true

    # Restore stable replicas
    scale_deployment "quantum-trader-stable" "${original_replicas}" || return 1
    wait_for_ready "quantum-trader-stable" "${original_replicas}" || return 1

    log_success "Rollback completed"
    return 0
}

################################################################################
# Main Canary Deployment Flow
################################################################################

main() {
    log_info "=== Quantum Trader AI Canary Deployment ==="
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
            -c|--canary)
                CANARY_PERCENT="$2"
                shift 2
                ;;
            -i|--interval)
                MONITOR_INTERVAL="$2"
                shift 2
                ;;
            -d|--duration)
                MONITOR_DURATION="$2"
                shift 2
                ;;
            -t|--threshold)
                ERROR_THRESHOLD="$2"
                shift 2
                ;;
            -e|--env)
                ENVIRONMENT="$2"
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

    # Validate inputs
    if [[ -z ${VERSION} ]]; then
        log_error "Version is required"
        show_help
        exit 1
    fi

    if [[ ${CANARY_PERCENT} -lt 1 || ${CANARY_PERCENT} -gt 50 ]]; then
        log_error "Canary percentage must be between 1 and 50"
        exit 1
    fi

    # Execute canary deployment
    verify_version_exists || exit 1
    create_canary_deployment || exit 1

    if monitor_canary; then
        promote_canary || exit 1
    else
        rollback_canary || exit 1
        exit 1
    fi

    log_info ""
    log_success "=== Canary Deployment Complete ==="
    log_info "Version: ${VERSION}"
    log_info "Environment: ${ENVIRONMENT}"

    exit 0
}

# Run main function
main "$@"
