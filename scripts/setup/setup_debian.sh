#!/usr/bin/env bash
# Debian/Ubuntu System Setup Script
# Installs system dependencies and configures the environment

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Info message
info() {
    echo -e "${BLUE}ℹ $@${NC}"
    log "INFO" "$@"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root. Use 'sudo' to run this script."
    fi
    success "Running with root privileges"
}

# Detect distribution
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        echo "$ID"
    else
        error_exit "Cannot detect Linux distribution"
    fi
}

# Update system packages
update_system() {
    log "INFO" "Updating system packages..."

    apt-get update
    apt-get upgrade -y

    success "System packages updated"
}

# Install essential build tools
install_build_tools() {
    log "INFO" "Installing essential build tools..."

    local packages=(
        "build-essential"
        "cmake"
        "git"
        "curl"
        "wget"
        "ca-certificates"
        "gnupg"
        "lsb-release"
        "apt-transport-https"
    )

    apt-get install -y "${packages[@]}"

    success "Essential build tools installed"
}

# Install development libraries
install_dev_libraries() {
    log "INFO" "Installing development libraries..."

    local packages=(
        "libssl-dev"
        "libffi-dev"
        "libpq-dev"
        "libmysqlclient-dev"
        "libbz2-dev"
        "liblzma-dev"
        "libsqlite3-dev"
        "libreadline-dev"
        "libncurses-dev"
        "zlib1g-dev"
    )

    apt-get install -y "${packages[@]}"

    success "Development libraries installed"
}

# Install system tools
install_system_tools() {
    log "INFO" "Installing system tools..."

    local packages=(
        "htop"
        "tmux"
        "vim"
        "nano"
        "git"
        "jq"
        "wget"
        "curl"
        "netcat-openbsd"
        "dnsutils"
        "iputils-ping"
        "net-tools"
        "traceroute"
        "telnet"
        "openssh-client"
        "openssh-server"
        "supervisor"
    )

    apt-get install -y "${packages[@]}"

    success "System tools installed"
}

# Install database clients
install_database_clients() {
    log "INFO" "Installing database clients..."

    local packages=(
        "postgresql-client"
        "mysql-client"
        "sqlite3"
        "redis-tools"
    )

    apt-get install -y "${packages[@]}"

    success "Database clients installed"
}

# Install monitoring tools
install_monitoring_tools() {
    log "INFO" "Installing monitoring tools..."

    local packages=(
        "sysstat"
        "iotop"
        "nethogs"
        "lsof"
        "strace"
        "perf-tools-unstable"
    )

    apt-get install -y "${packages[@]}" || warning "Some monitoring tools could not be installed"

    success "Monitoring tools installed"
}

# Setup firewall
setup_firewall() {
    log "INFO" "Setting up firewall (UFW)..."

    apt-get install -y ufw

    # Enable firewall
    echo "y" | ufw enable 2>/dev/null || warning "Could not enable UFW"

    # Allow SSH
    ufw allow ssh || warning "Could not allow SSH"

    # Allow common ports
    ufw allow 8000/tcp || warning "Could not allow port 8000"
    ufw allow 8080/tcp || warning "Could not allow port 8080"
    ufw allow 443/tcp || warning "Could not allow port 443"
    ufw allow 80/tcp || warning "Could not allow port 80"

    success "Firewall configured"
}

# Configure system limits
configure_system_limits() {
    log "INFO" "Configuring system limits..."

    # Increase file descriptors limit
    cat >> /etc/security/limits.conf <<EOF

# Quantum Trader AI settings
* soft nofile 65535
* hard nofile 65535
* soft nproc 32768
* hard nproc 32768
EOF

    # Increase network limits
    cat >> /etc/sysctl.conf <<EOF

# Quantum Trader AI network settings
net.core.somaxconn=4096
net.ipv4.tcp_max_syn_backlog=4096
net.ipv4.ip_local_port_range=1024 65535
net.core.netdev_max_backlog=5000
EOF

    sysctl -p >/dev/null 2>&1 || warning "Could not apply sysctl settings"

    success "System limits configured"
}

# Setup swap space
setup_swap() {
    log "INFO" "Setting up swap space..."

    local swap_size="4G"

    if grep -q "swapfile" /etc/fstab; then
        log "INFO" "Swap already configured, skipping setup"
        return 0
    fi

    # Create swap file
    fallocate -l "$swap_size" /swapfile || error_exit "Could not allocate swap space"
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile

    # Make permanent
    echo "/swapfile none swap sw 0 0" >> /etc/fstab

    success "Swap space configured ($swap_size)"
}

# Install timezone/time sync tools
install_time_tools() {
    log "INFO" "Installing time synchronization tools..."

    apt-get install -y chrony ntp

    success "Time synchronization tools installed"
}

# Create project directories
create_directories() {
    log "INFO" "Creating project directories..."

    mkdir -p "$PROJECT_ROOT"/{logs,data,backups,deployments,.cache}

    # Set proper permissions
    chmod 755 "$PROJECT_ROOT"/{logs,data,backups,deployments}

    success "Project directories created"
}

# Final checks
run_final_checks() {
    log "INFO" "Running final checks..."

    info "System information:"
    echo "  OS: $(lsb_release -ds)"
    echo "  Kernel: $(uname -r)"
    echo "  CPU cores: $(nproc)"
    echo "  Memory: $(free -h | awk 'NR==2 {print $2}')"
    echo "  Disk: $(df -h / | awk 'NR==2 {print $2}')"

    success "Setup completed successfully!"
}

# Main execution
main() {
    log "INFO" "=== Debian/Ubuntu System Setup Started ==="

    check_root

    local distro
    distro=$(detect_distro)

    if [[ ! "$distro" =~ ^(debian|ubuntu)$ ]]; then
        warning "This script is optimized for Debian/Ubuntu. Detected: $distro"
    fi

    log "INFO" "Detected distribution: $distro"

    update_system
    install_build_tools
    install_dev_libraries
    install_system_tools
    install_database_clients
    install_monitoring_tools
    setup_firewall
    configure_system_limits
    setup_swap
    install_time_tools
    create_directories
    run_final_checks

    log "INFO" "=== Debian/Ubuntu System Setup Completed ==="
}

# Run main function
main "$@"
