#!/usr/bin/env bash

################################################################################
# Quantum Trader AI Backup Script
#
# This script performs comprehensive backups of the Quantum Trader AI system,
# including databases, configurations, and trading data. Supports local and
# remote backup destinations with encryption and compression.
#
# Usage: ./backup.sh [OPTIONS]
#
# Options:
#   -h, --help              Show this help message
#   -t, --type TYPE         Backup type (full/incremental/database, default: full)
#   -d, --destination DIR   Backup destination (default: ./backups)
#   -r, --remote HOST:PATH  Remote backup destination (SSH)
#   -c, --compress          Compress backup files (default: yes)
#   -e, --encrypt           Encrypt backup files (requires encryption key)
#   -k, --keep-days DAYS    Keep backups for N days (default: 30)
#   -v, --verbose           Enable verbose output
#
# Examples:
#   ./backup.sh --type full --destination /backups/quantum_trader
#   ./backup.sh --type database --remote backup.example.com:/backups
#   ./backup.sh --type incremental --encrypt --keep-days 60
#
################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BACKUP_TYPE="${BACKUP_TYPE:-full}"
BACKUP_DEST="${BACKUP_DEST:-${PROJECT_ROOT}/backups}"
REMOTE_DEST="${REMOTE_DEST:-}"
COMPRESS="${COMPRESS:-1}"
ENCRYPT="${ENCRYPT:-0}"
ENCRYPTION_KEY="${ENCRYPTION_KEY:-}"
KEEP_DAYS="${KEEP_DAYS:-30}"
VERBOSE="${VERBOSE:-0}"

# Database configuration
DB_HOST="${DATABASE_HOST:-localhost}"
DB_PORT="${DATABASE_PORT:-5432}"
DB_NAME="${DATABASE_NAME:-quantum_trader}"
DB_USER="${DATABASE_USER:-postgres}"

# Backup variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_DEST}/backup_${TIMESTAMP}"
BACKUP_NAME="quantum_trader_${BACKUP_TYPE}_${TIMESTAMP}"

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
    head -n 28 "$0" | tail -n +2 | sed 's/^# //'
}

get_size() {
    du -sh "$1" 2>/dev/null | cut -f1 || echo "unknown"
}

################################################################################
# Database Backup Functions
################################################################################

backup_database() {
    log_info "Starting database backup..."

    mkdir -p "${BACKUP_DIR}"
    local db_backup="${BACKUP_DIR}/database_dump.sql"

    if PGPASSWORD="${DATABASE_PASSWORD:-}" pg_dump \
        -h "${DB_HOST}" -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --no-password \
        -Fc -f "${db_backup}"; then

        local db_size
        db_size=$(get_size "${db_backup}")
        log_success "Database backup completed (${db_size})"
        return 0
    else
        log_error "Database backup failed"
        return 1
    fi
}

backup_application_data() {
    log_info "Backing up application data..."

    local data_backup="${BACKUP_DIR}/app_data"
    mkdir -p "${data_backup}"

    # Backup trading data
    if [[ -d ${PROJECT_ROOT}/data ]]; then
        cp -r "${PROJECT_ROOT}/data" "${data_backup}/trading_data" 2>/dev/null || true
    fi

    # Backup configuration files
    if [[ -f ${PROJECT_ROOT}/.env ]]; then
        cp "${PROJECT_ROOT}/.env" "${data_backup}/.env.backup" || log_warning "Could not backup .env"
    fi

    # Backup model files
    if [[ -d ${PROJECT_ROOT}/models ]]; then
        cp -r "${PROJECT_ROOT}/models" "${data_backup}/models" 2>/dev/null || true
    fi

    log_success "Application data backup completed"
    return 0
}

################################################################################
# File Backup Functions
################################################################################

backup_configuration() {
    log_info "Backing up configuration files..."

    local config_backup="${BACKUP_DIR}/config"
    mkdir -p "${config_backup}"

    # Backup configuration directory
    if [[ -d ${PROJECT_ROOT}/config ]]; then
        cp -r "${PROJECT_ROOT}/config"/* "${config_backup}/" 2>/dev/null || true
    fi

    # Backup scripts
    if [[ -d ${PROJECT_ROOT}/scripts ]]; then
        cp -r "${PROJECT_ROOT}/scripts" "${config_backup}/" 2>/dev/null || true
    fi

    log_success "Configuration backup completed"
    return 0
}

backup_full() {
    log_info "Starting full backup..."

    backup_database || return 1
    backup_application_data || return 1
    backup_configuration || return 1

    log_success "Full backup completed"
    return 0
}

backup_incremental() {
    log_info "Starting incremental backup..."

    # Find files modified in last backup interval (typically 24 hours)
    local last_backup
    last_backup=$(find "${BACKUP_DEST}" -maxdepth 1 -type d -name "backup_*" | sort -r | head -n 2 | tail -n 1)

    mkdir -p "${BACKUP_DIR}"

    if [[ -n ${last_backup} && -d ${last_backup} ]]; then
        local newer_than
        newer_than=$(stat -c %Y "${last_backup}")

        # Find files newer than last backup
        find "${PROJECT_ROOT}" \
            -type f \
            -newer "${last_backup}" \
            -not -path "*/\.*" \
            2>/dev/null | while read -r file; do
            local rel_path
            rel_path="${file#${PROJECT_ROOT}/}"
            mkdir -p "$(dirname "${BACKUP_DIR}/${rel_path}")"
            cp "${file}" "${BACKUP_DIR}/${rel_path}" 2>/dev/null || true
        done

        log_success "Incremental backup completed"
    else
        log_info "No previous backup found, performing full backup instead"
        backup_full || return 1
    fi

    return 0
}

################################################################################
# Compression & Encryption Functions
################################################################################

compress_backup() {
    log_info "Compressing backup..."

    local archive_name="${BACKUP_DEST}/${BACKUP_NAME}.tar.gz"

    if tar -czf "${archive_name}" -C "${BACKUP_DEST}" "backup_${TIMESTAMP}"; then
        local size
        size=$(get_size "${archive_name}")
        log_success "Backup compressed (${size})"

        # Remove uncompressed directory
        rm -rf "${BACKUP_DIR}"
        echo "${archive_name}"
        return 0
    else
        log_error "Compression failed"
        return 1
    fi
}

encrypt_backup() {
    local backup_file=$1

    log_info "Encrypting backup..."

    if [[ -z ${ENCRYPTION_KEY} ]]; then
        log_error "Encryption key not provided"
        return 1
    fi

    local encrypted_file="${backup_file}.enc"

    if openssl enc -aes-256-cbc -salt -in "${backup_file}" \
        -out "${encrypted_file}" -k "${ENCRYPTION_KEY}"; then

        log_success "Backup encrypted"
        rm -f "${backup_file}"
        echo "${encrypted_file}"
        return 0
    else
        log_error "Encryption failed"
        return 1
    fi
}

################################################################################
# Remote Backup Functions
################################################################################

transfer_to_remote() {
    local backup_file=$1
    local remote=$2

    log_info "Transferring backup to remote: ${remote}..."

    local remote_host="${remote%:*}"
    local remote_path="${remote#*:}"

    if scp -P 22 "${backup_file}" "${remote_host}:${remote_path}/" 2>/dev/null; then
        log_success "Backup transferred to remote"
        return 0
    else
        log_error "Remote transfer failed"
        return 1
    fi
}

################################################################################
# Cleanup & Maintenance Functions
################################################################################

cleanup_old_backups() {
    log_info "Cleaning up old backups..."

    local cutoff_date
    cutoff_date=$(date -d "${KEEP_DAYS} days ago" +%Y%m%d)

    find "${BACKUP_DEST}" -maxdepth 1 -type d -name "backup_*" | while read -r backup_dir; do
        local backup_date
        backup_date=$(basename "${backup_dir}" | sed 's/backup_//' | cut -d_ -f1)

        if [[ ${backup_date} -lt ${cutoff_date} ]]; then
            log_info "Removing old backup: ${backup_dir}"
            rm -rf "${backup_dir}"
        fi
    done

    # Also cleanup compressed backups
    find "${BACKUP_DEST}" -maxdepth 1 -type f -name "*.tar.gz" | while read -r backup_file; do
        local backup_date
        backup_date=$(basename "${backup_file}" | sed 's/quantum_trader_[^_]*_//' | sed 's/\.tar\.gz.*//' | cut -d_ -f1)

        if [[ ${backup_date} -lt ${cutoff_date} ]]; then
            log_info "Removing old backup: ${backup_file}"
            rm -f "${backup_file}"
        fi
    done

    log_success "Old backups cleaned up"
    return 0
}

generate_backup_report() {
    log_info ""
    log_info "=== Backup Report ==="
    log_info "Backup Type: ${BACKUP_TYPE}"
    log_info "Backup Time: ${TIMESTAMP}"
    log_info "Backup Location: ${BACKUP_DEST}"

    if [[ -f ${BACKUP_DEST}/${BACKUP_NAME}.tar.gz ]]; then
        local size
        size=$(get_size "${BACKUP_DEST}/${BACKUP_NAME}.tar.gz")
        log_success "Final Backup Size: ${size}"
    fi

    # Count backups
    local backup_count
    backup_count=$(find "${BACKUP_DEST}" -maxdepth 1 \( -type d -name "backup_*" -o -type f -name "*.tar.gz" \) | wc -l)
    log_info "Total Backups Stored: ${backup_count}"

    return 0
}

################################################################################
# Main Backup Flow
################################################################################

main() {
    log_info "=== Quantum Trader AI Backup System ==="
    log_info ""

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--type)
                BACKUP_TYPE="$2"
                shift 2
                ;;
            -d|--destination)
                BACKUP_DEST="$2"
                shift 2
                ;;
            -r|--remote)
                REMOTE_DEST="$2"
                shift 2
                ;;
            -c|--compress)
                COMPRESS=1
                shift
                ;;
            -e|--encrypt)
                ENCRYPT=1
                shift
                ;;
            -k|--keep-days)
                KEEP_DAYS="$2"
                shift 2
                ;;
            -v|--verbose)
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

    # Create backup directory
    mkdir -p "${BACKUP_DEST}"

    # Perform backup based on type
    case ${BACKUP_TYPE} in
        full)
            backup_full || exit 1
            ;;
        database)
            backup_database || exit 1
            ;;
        incremental)
            backup_incremental || exit 1
            ;;
        *)
            log_error "Unknown backup type: ${BACKUP_TYPE}"
            exit 1
            ;;
    esac

    # Process backup file
    local backup_file="${BACKUP_DIR}"
    if [[ ${COMPRESS} -eq 1 ]]; then
        backup_file=$(compress_backup) || exit 1
    fi

    if [[ ${ENCRYPT} -eq 1 ]]; then
        backup_file=$(encrypt_backup "${backup_file}") || exit 1
    fi

    # Transfer to remote if specified
    if [[ -n ${REMOTE_DEST} ]]; then
        transfer_to_remote "${backup_file}" "${REMOTE_DEST}" || exit 1
    fi

    # Cleanup old backups
    cleanup_old_backups || exit 1

    # Generate report
    generate_backup_report

    log_info ""
    log_success "=== Backup Complete ==="

    exit 0
}

# Run main function
main "$@"
