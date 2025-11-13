#!/usr/bin/env bash

################################################################################
# Quantum Trader AI Cleanup Script
#
# This script performs system maintenance and cleanup tasks, including removing
# temporary files, cleaning logs, clearing caches, and optimizing storage.
#
# Usage: ./cleanup.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help message
#   -t, --type TYPE         Cleanup type (all/logs/cache/temp/docker, default: all)
#   -d, --days DAYS         Remove files older than N days (default: 30)
#   -f, --force             Force cleanup without confirmation
#   -r, --dry-run           Show what would be removed without doing it
#   --verbose               Enable verbose output
#
# Cleanup Types:
#   - logs              Clean application logs
#   - cache             Clear application caches
#   - temp              Remove temporary files
#   - docker            Clean Docker images and containers
#   - all               Perform all cleanup operations
#
# Examples:
#   ./cleanup.sh --type all
#   ./cleanup.sh --type logs --days 7 --force
#   ./cleanup.sh --type docker --dry-run
#
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLEANUP_TYPE="${CLEANUP_TYPE:-all}"
DAYS_OLD="${DAYS_OLD:-30}"
FORCE_CLEANUP="${FORCE_CLEANUP:-0}"
DRY_RUN="${DRY_RUN:-0}"
VERBOSE="${VERBOSE:-0}"

# Tracking
TOTAL_FREED=0
FILES_REMOVED=0

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
    head -n 34 "$0" | tail -n +2 | sed 's/^# //'
}

confirm() {
    local prompt=$1

    if [[ ${FORCE_CLEANUP} -eq 1 ]]; then
        return 0
    fi

    read -p "$(echo -e "${YELLOW}${prompt} (y/n)${NC} ")" -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

remove_file_safe() {
    local file=$1

    if [[ ${DRY_RUN} -eq 1 ]]; then
        verbose_log "[DRY RUN] Would remove: ${file}"
        return 0
    fi

    if rm -f "${file}" 2>/dev/null; then
        FILES_REMOVED=$((FILES_REMOVED + 1))
        verbose_log "Removed: ${file}"
        return 0
    else
        return 1
    fi
}

get_size() {
    du -sh "$1" 2>/dev/null | cut -f1 || echo "unknown"
}

################################################################################
# Log Cleanup Functions
################################################################################

cleanup_logs() {
    log_info "Cleaning up logs (files older than ${DAYS_OLD} days)..."

    if ! confirm "Remove old log files?"; then
        log_warning "Log cleanup cancelled"
        return 0
    fi

    # Application logs
    if [[ -d ${PROJECT_ROOT}/logs ]]; then
        local log_count
        log_count=$(find "${PROJECT_ROOT}/logs" -type f -name "*.log" -mtime +${DAYS_OLD} | wc -l)

        if [[ ${log_count} -gt 0 ]]; then
            log_info "Removing ${log_count} old log files..."

            find "${PROJECT_ROOT}/logs" -type f -name "*.log" -mtime +${DAYS_OLD} | while read -r logfile; do
                remove_file_safe "${logfile}"
            done
        fi
    fi

    # System logs
    if [[ -d /var/log && -w /var/log ]]; then
        local syslog_count
        syslog_count=$(find /var/log -name "quantum-trader*.log" -mtime +${DAYS_OLD} 2>/dev/null | wc -l)

        if [[ ${syslog_count} -gt 0 ]]; then
            log_info "Removing ${syslog_count} old system log files..."

            find /var/log -name "quantum-trader*.log" -mtime +${DAYS_OLD} 2>/dev/null | while read -r logfile; do
                remove_file_safe "${logfile}" || true
            done
        fi
    fi

    log_success "Log cleanup completed"
    return 0
}

truncate_rotating_logs() {
    log_info "Truncating rotating log files..."

    if [[ -d ${PROJECT_ROOT}/logs ]]; then
        find "${PROJECT_ROOT}/logs" -name "*.log.*" -type f | while read -r logfile; do
            if [[ ${DRY_RUN} -eq 1 ]]; then
                verbose_log "[DRY RUN] Would truncate: ${logfile}"
            else
                : > "${logfile}"  # Truncate file
                verbose_log "Truncated: ${logfile}"
            fi
        done
    fi

    return 0
}

################################################################################
# Cache Cleanup Functions
################################################################################

cleanup_cache() {
    log_info "Clearing application caches..."

    if ! confirm "Clear application caches?"; then
        log_warning "Cache cleanup cancelled"
        return 0
    fi

    # Python cache
    if [[ -d ${PROJECT_ROOT}/__pycache__ ]]; then
        log_info "Removing Python cache files..."

        if [[ ${DRY_RUN} -eq 1 ]]; then
            verbose_log "[DRY RUN] Would remove: ${PROJECT_ROOT}/__pycache__"
        else
            find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
            find "${PROJECT_ROOT}" -type f -name "*.pyc" -delete 2>/dev/null || true
        fi
    fi

    # Redis cache (if accessible)
    if command -v redis-cli &> /dev/null; then
        log_info "Flushing Redis cache..."

        if redis-cli ping > /dev/null 2>&1; then
            if [[ ${DRY_RUN} -eq 1 ]]; then
                verbose_log "[DRY RUN] Would flush Redis cache"
            else
                redis-cli FLUSHDB > /dev/null 2>&1 || log_warning "Could not flush Redis"
            fi
        else
            log_warning "Redis is not available"
        fi
    fi

    # Temporary cache directories
    for cache_dir in /tmp/quantum_trader* ~/.cache/quantum_trader; do
        if [[ -d ${cache_dir} ]]; then
            log_info "Removing cache directory: ${cache_dir}"

            if [[ ${DRY_RUN} -eq 1 ]]; then
                verbose_log "[DRY RUN] Would remove: ${cache_dir}"
            else
                rm -rf "${cache_dir}" 2>/dev/null || true
            fi
        fi
    done

    log_success "Cache cleanup completed"
    return 0
}

################################################################################
# Temporary File Cleanup Functions
################################################################################

cleanup_temp_files() {
    log_info "Removing temporary files..."

    if ! confirm "Remove temporary files?"; then
        log_warning "Temporary file cleanup cancelled"
        return 0
    fi

    # Application temp files
    local temp_patterns=("*.tmp" "*.temp" "*.bak" "*~")

    for pattern in "${temp_patterns[@]}"; do
        log_info "Removing ${pattern} files..."

        find "${PROJECT_ROOT}" -type f -name "${pattern}" -delete 2>/dev/null || true
    done

    # System temp files
    for tempdir in /tmp /var/tmp; do
        if [[ -d ${tempdir} && -w ${tempdir} ]]; then
            log_info "Cleaning ${tempdir}..."

            find "${tempdir}" -type f -mtime +${DAYS_OLD} -delete 2>/dev/null || true
        fi
    done

    # Build artifacts
    if [[ -d ${PROJECT_ROOT}/build ]]; then
        log_info "Removing build artifacts..."

        if [[ ${DRY_RUN} -eq 1 ]]; then
            verbose_log "[DRY RUN] Would remove: ${PROJECT_ROOT}/build"
        else
            rm -rf "${PROJECT_ROOT}/build"
        fi
    fi

    if [[ -d ${PROJECT_ROOT}/dist ]]; then
        log_info "Removing distribution artifacts..."

        if [[ ${DRY_RUN} -eq 1 ]]; then
            verbose_log "[DRY RUN] Would remove old dist files"
        else
            find "${PROJECT_ROOT}/dist" -type f -mtime +${DAYS_OLD} -delete 2>/dev/null || true
        fi
    fi

    log_success "Temporary file cleanup completed"
    return 0
}

################################################################################
# Docker Cleanup Functions
################################################################################

cleanup_docker() {
    log_info "Cleaning Docker resources..."

    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed, skipping Docker cleanup"
        return 0
    fi

    if ! confirm "Remove unused Docker images and containers?"; then
        log_warning "Docker cleanup cancelled"
        return 0
    fi

    # Remove stopped containers
    log_info "Removing stopped containers..."

    if [[ ${DRY_RUN} -eq 1 ]]; then
        verbose_log "[DRY RUN] Would remove stopped containers"
        docker ps -a --filter status=exited -q 2>/dev/null | while read -r container; do
            verbose_log "[DRY RUN] Container: ${container}"
        done
    else
        docker container prune -f 2>/dev/null || log_warning "Failed to prune containers"
    fi

    # Remove dangling images
    log_info "Removing dangling images..."

    if [[ ${DRY_RUN} -eq 1 ]]; then
        verbose_log "[DRY RUN] Would remove dangling images"
        docker images -f dangling=true -q 2>/dev/null | while read -r image; do
            verbose_log "[DRY RUN] Image: ${image}"
        done
    else
        docker image prune -f 2>/dev/null || log_warning "Failed to prune images"
    fi

    # Remove unused volumes
    log_info "Removing unused volumes..."

    if [[ ${DRY_RUN} -eq 1 ]]; then
        verbose_log "[DRY RUN] Would remove unused volumes"
    else
        docker volume prune -f 2>/dev/null || log_warning "Failed to prune volumes"
    fi

    log_success "Docker cleanup completed"
    return 0
}

################################################################################
# Database Optimization Functions
################################################################################

optimize_database() {
    log_info "Optimizing database..."

    local db_host="${DATABASE_HOST:-localhost}"
    local db_port="${DATABASE_PORT:-5432}"
    local db_name="${DATABASE_NAME:-quantum_trader}"
    local db_user="${DATABASE_USER:-postgres}"

    if ! command -v psql &> /dev/null; then
        log_warning "PostgreSQL client is not installed, skipping database optimization"
        return 0
    fi

    if ! confirm "Run database optimization (VACUUM ANALYZE)?"; then
        log_warning "Database optimization cancelled"
        return 0
    fi

    log_info "Running VACUUM ANALYZE..."

    if PGPASSWORD="${DATABASE_PASSWORD:-}" psql -h "${db_host}" -p "${db_port}" \
        -U "${db_user}" -d "${db_name}" -c "VACUUM ANALYZE;" 2>/dev/null; then
        log_success "Database optimized"
    else
        log_warning "Database optimization encountered issues"
    fi

    return 0
}

################################################################################
# System Statistics Functions
################################################################################

show_disk_usage() {
    log_info ""
    log_info "=== Disk Usage Summary ==="

    local before_size
    before_size=$(get_size "${PROJECT_ROOT}")
    log_info "Project directory size: ${before_size}"

    if [[ -d ${PROJECT_ROOT}/logs ]]; then
        local logs_size
        logs_size=$(get_size "${PROJECT_ROOT}/logs")
        log_info "Logs directory size: ${logs_size}"
    fi

    if [[ -d ${PROJECT_ROOT}/backups ]]; then
        local backups_size
        backups_size=$(get_size "${PROJECT_ROOT}/backups")
        log_info "Backups directory size: ${backups_size}"
    fi

    return 0
}

################################################################################
# Main Cleanup Flow
################################################################################

main() {
    log_info "=== Quantum Trader AI System Cleanup ==="
    log_info ""

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--type)
                CLEANUP_TYPE="$2"
                shift 2
                ;;
            -d|--days)
                DAYS_OLD="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_CLEANUP=1
                shift
                ;;
            -r|--dry-run)
                DRY_RUN=1
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

    # Show dry-run notice
    if [[ ${DRY_RUN} -eq 1 ]]; then
        log_warning "DRY RUN MODE - No files will be removed"
    fi

    # Execute cleanup based on type
    case ${CLEANUP_TYPE} in
        all)
            cleanup_logs || true
            truncate_rotating_logs || true
            cleanup_cache || true
            cleanup_temp_files || true
            cleanup_docker || true
            optimize_database || true
            ;;
        logs)
            cleanup_logs || true
            truncate_rotating_logs || true
            ;;
        cache)
            cleanup_cache || true
            ;;
        temp)
            cleanup_temp_files || true
            ;;
        docker)
            cleanup_docker || true
            ;;
        *)
            log_error "Unknown cleanup type: ${CLEANUP_TYPE}"
            show_help
            exit 1
            ;;
    esac

    # Show summary
    show_disk_usage

    log_info ""
    log_success "=== Cleanup Complete ==="

    exit 0
}

# Run main function
main "$@"
