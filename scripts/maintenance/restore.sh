#!/usr/bin/env bash
# Database and System Restore Script
# Restores database from backup and recovers system state

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/restore_$(date +%Y%m%d_%H%M%S).log"
BACKUP_DIR="${BACKUP_DIR:-${PROJECT_ROOT}/backups}"
RESTORE_TIMESTAMP="${1:-}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."

    if [[ ! -d "$BACKUP_DIR" ]]; then
        error_exit "Backup directory not found: $BACKUP_DIR"
    fi

    # Check for required commands
    local required_commands=("docker" "docker-compose" "pg_restore")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "Required command not found: $cmd"
        fi
    done

    success "Prerequisites check passed"
}

# Find backup file
find_backup() {
    log "INFO" "Looking for backup file..."

    local backup_file
    if [[ -n "$RESTORE_TIMESTAMP" ]]; then
        backup_file="${BACKUP_DIR}/database_${RESTORE_TIMESTAMP}.sql.gz"
        if [[ ! -f "$backup_file" ]]; then
            error_exit "Backup file not found: $backup_file"
        fi
    else
        # Find the most recent backup
        backup_file=$(find "$BACKUP_DIR" -name "database_*.sql.gz" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
        if [[ -z "$backup_file" ]]; then
            error_exit "No backup files found in $BACKUP_DIR"
        fi
    fi

    log "INFO" "Using backup file: $backup_file"
    echo "$backup_file"
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"

    log "INFO" "Verifying backup file integrity..."

    if ! gzip -t "$backup_file" 2>/dev/null; then
        error_exit "Backup file is corrupted: $backup_file"
    fi

    success "Backup file integrity verified"
}

# Stop services
stop_services() {
    log "INFO" "Stopping services..."

    cd "$PROJECT_ROOT"
    if docker-compose ps --services --filter "status=running" | grep -q .; then
        docker-compose down --timeout=30 2>/dev/null || warning "Could not gracefully stop services"
    fi

    success "Services stopped"
}

# Restore database
restore_database() {
    local backup_file="$1"

    log "INFO" "Restoring database from backup..."

    # Start PostgreSQL container
    cd "$PROJECT_ROOT"
    docker-compose up -d postgres 2>&1 | tee -a "$LOG_FILE"

    # Wait for PostgreSQL to be ready
    local max_attempts=30
    local attempt=0
    while ! docker-compose exec -T postgres pg_isready -U postgres &>/dev/null; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            error_exit "PostgreSQL failed to start within timeout"
        fi
        sleep 1
    done

    success "PostgreSQL is ready"

    # Get database name and credentials from environment
    local db_name="${DB_NAME:-quantum_trader}"
    local db_user="${DB_USER:-postgres}"

    # Create database if it doesn't exist
    docker-compose exec -T postgres createdb -U "$db_user" "$db_name" 2>/dev/null || true

    # Restore database
    log "INFO" "Restoring data from backup..."
    gunzip -c "$backup_file" | docker-compose exec -T postgres psql -U "$db_user" "$db_name" 2>&1 | tee -a "$LOG_FILE"

    success "Database restored successfully"
}

# Verify restoration
verify_restoration() {
    log "INFO" "Verifying database restoration..."

    cd "$PROJECT_ROOT"
    local db_name="${DB_NAME:-quantum_trader}"
    local db_user="${DB_USER:-postgres}"

    # Check table count
    local table_count=$(docker-compose exec -T postgres psql -U "$db_user" "$db_name" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'" 2>/dev/null || echo "0")

    if [[ "$table_count" -gt 0 ]]; then
        success "Database restoration verified ($table_count tables)"
    else
        warning "No tables found in restored database"
    fi
}

# Start services
start_services() {
    log "INFO" "Starting services..."

    cd "$PROJECT_ROOT"
    docker-compose up -d 2>&1 | tee -a "$LOG_FILE"

    # Wait for services to be ready
    sleep 10

    success "Services started"
}

# Main execution
main() {
    log "INFO" "=== Database and System Restore Started ==="
    log "INFO" "Backup directory: $BACKUP_DIR"
    log "INFO" "Project root: $PROJECT_ROOT"

    check_prerequisites

    local backup_file
    backup_file=$(find_backup)

    verify_backup "$backup_file"

    stop_services

    restore_database "$backup_file"

    verify_restoration

    start_services

    log "INFO" "=== Database and System Restore Completed Successfully ==="
    success "Restore operation completed! Check logs at $LOG_FILE"
}

# Run main function
main "$@"
