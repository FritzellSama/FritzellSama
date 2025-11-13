#!/usr/bin/env bash
# Log Rotation Script
# Rotates log files, compresses old logs, and cleans up expired logs

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOGS_DIR="${LOGS_DIR:-${PROJECT_ROOT}/logs}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${LOGS_DIR}/.archive}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
MAX_FILE_SIZE="${MAX_FILE_SIZE:-104857600}" # 100MB default

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."

    if [[ ! -d "$LOGS_DIR" ]]; then
        error_exit "Logs directory not found: $LOGS_DIR"
    fi

    # Check for required commands
    local required_commands=("find" "gzip" "date" "du")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "Required command not found: $cmd"
        fi
    done

    success "Prerequisites check passed"
}

# Create archive directory
create_archive_dir() {
    if [[ ! -d "$ARCHIVE_DIR" ]]; then
        mkdir -p "$ARCHIVE_DIR"
        log "INFO" "Created archive directory: $ARCHIVE_DIR"
    fi
}

# Rotate single log file
rotate_log_file() {
    local log_file="$1"
    local file_size

    file_size=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null || echo 0)

    if [[ $file_size -gt $MAX_FILE_SIZE ]]; then
        local timestamp
        timestamp=$(date +%Y%m%d_%H%M%S)
        local rotated_file="${log_file}.${timestamp}"

        log "INFO" "Rotating: $log_file ($(numfmt --to=iec-i --suffix=B "$file_size" 2>/dev/null || echo "${file_size} bytes"))"

        # Move log file
        mv "$log_file" "$rotated_file"

        # Create new empty log file with same permissions
        touch "$log_file"
        chmod --reference="$rotated_file" "$log_file" 2>/dev/null || true

        # Compress rotated file
        log "INFO" "Compressing: $rotated_file"
        gzip "$rotated_file"

        # Move to archive
        mv "${rotated_file}.gz" "$ARCHIVE_DIR/"

        success "Rotated: $(basename "$log_file")"
        return 0
    fi

    return 1
}

# Rotate all log files
rotate_all_logs() {
    log "INFO" "Starting log rotation..."

    local rotated_count=0

    if [[ ! -d "$LOGS_DIR" ]]; then
        warning "Logs directory not found: $LOGS_DIR"
        return 1
    fi

    while IFS= read -r -d '' log_file; do
        if rotate_log_file "$log_file"; then
            rotated_count=$((rotated_count + 1))
        fi
    done < <(find "$LOGS_DIR" -maxdepth 1 -name "*.log" -type f -print0 2>/dev/null)

    if [[ $rotated_count -gt 0 ]]; then
        success "Rotated $rotated_count log files"
    else
        log "INFO" "No log files needed rotation"
    fi
}

# Clean up old archives
cleanup_old_archives() {
    log "INFO" "Cleaning up old archives (retention: $RETENTION_DAYS days)..."

    if [[ ! -d "$ARCHIVE_DIR" ]]; then
        log "INFO" "Archive directory does not exist, skipping cleanup"
        return 0
    fi

    local cleanup_count=0

    while IFS= read -r -d '' archive_file; do
        log "INFO" "Removing expired archive: $(basename "$archive_file")"
        rm -f "$archive_file"
        cleanup_count=$((cleanup_count + 1))
    done < <(find "$ARCHIVE_DIR" -name "*.log.*.gz" -type f -mtime +"$RETENTION_DAYS" -print0 2>/dev/null)

    if [[ $cleanup_count -gt 0 ]]; then
        success "Removed $cleanup_count expired archive files"
    else
        log "INFO" "No expired archives found"
    fi
}

# Generate archive report
generate_report() {
    log "INFO" "Generating log rotation report..."

    echo ""
    echo "=== Log Rotation Report ==="
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    echo "Current Logs Directory:"
    du -sh "$LOGS_DIR" 2>/dev/null || echo "  Unable to calculate size"

    if [[ -d "$ARCHIVE_DIR" ]]; then
        echo ""
        echo "Archive Directory:"
        du -sh "$ARCHIVE_DIR" 2>/dev/null || echo "  Unable to calculate size"

        echo ""
        echo "Top 10 Largest Archives:"
        find "$ARCHIVE_DIR" -name "*.log.*.gz" -type f -exec ls -lh {} \; | \
            sort -k5 -hr | head -10 | awk '{print "  " $9 " (" $5 ")"}'
    fi

    echo ""
    echo "Log Files Statistics:"
    find "$LOGS_DIR" -maxdepth 1 -name "*.log" -type f -exec ls -lh {} \; | \
        awk '{print "  " $9 " (" $5 ")"}' || echo "  No log files found"

    echo ""
}

# Validate rotation settings
validate_settings() {
    log "INFO" "Validating rotation settings..."

    if [[ ! "$RETENTION_DAYS" =~ ^[0-9]+$ ]]; then
        error_exit "Invalid RETENTION_DAYS: $RETENTION_DAYS (must be a number)"
    fi

    if [[ ! "$MAX_FILE_SIZE" =~ ^[0-9]+$ ]]; then
        error_exit "Invalid MAX_FILE_SIZE: $MAX_FILE_SIZE (must be a number)"
    fi

    log "INFO" "Retention days: $RETENTION_DAYS"
    log "INFO" "Max file size: $MAX_FILE_SIZE bytes"
    success "Settings validated"
}

# Main execution
main() {
    log "INFO" "=== Log Rotation Started ==="
    log "INFO" "Logs directory: $LOGS_DIR"
    log "INFO" "Archive directory: $ARCHIVE_DIR"

    check_prerequisites
    validate_settings
    create_archive_dir

    rotate_all_logs

    cleanup_old_archives

    generate_report

    log "INFO" "=== Log Rotation Completed Successfully ==="
    success "Log rotation completed!"
}

# Run main function
main "$@"
