#!/usr/bin/env bash
# Docker Installation Script
# Installs Docker and Docker Compose on the system

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_VERSION="${DOCKER_VERSION:-latest}"
DOCKER_COMPOSE_VERSION="${DOCKER_COMPOSE_VERSION:-latest}"

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

# Uninstall old Docker versions
uninstall_old_docker() {
    log "INFO" "Removing old Docker versions..."

    apt-get remove -y docker docker.io docker-engine docker-compose 2>/dev/null || true

    success "Old Docker versions removed"
}

# Add Docker GPG key
add_docker_gpg_key() {
    log "INFO" "Adding Docker GPG key..."

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    success "Docker GPG key added"
}

# Setup Docker repository
setup_docker_repository() {
    log "INFO" "Setting up Docker repository..."

    local distro
    distro=$(detect_distro)

    case "$distro" in
        ubuntu)
            local arch
            arch=$(dpkg --print-architecture)

            echo "deb [arch=${arch} signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
                tee /etc/apt/sources.list.d/docker.list > /dev/null
            ;;
        debian)
            local arch
            arch=$(dpkg --print-architecture)

            echo "deb [arch=${arch} signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | \
                tee /etc/apt/sources.list.d/docker.list > /dev/null
            ;;
        *)
            error_exit "Unsupported distribution: $distro"
            ;;
    esac

    apt-get update

    success "Docker repository configured"
}

# Install Docker
install_docker() {
    log "INFO" "Installing Docker..."

    if [[ "$DOCKER_VERSION" == "latest" ]]; then
        apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    else
        apt-get install -y "docker-ce=${DOCKER_VERSION}" "docker-ce-cli=${DOCKER_VERSION}" containerd.io
    fi

    success "Docker installed"
}

# Verify Docker installation
verify_docker_installation() {
    log "INFO" "Verifying Docker installation..."

    docker --version || error_exit "Docker installation verification failed"
    docker run --rm hello-world > /dev/null || error_exit "Docker runtime test failed"

    success "Docker installation verified"
}

# Enable Docker service
enable_docker_service() {
    log "INFO" "Enabling Docker service..."

    systemctl enable docker
    systemctl start docker

    # Wait for Docker daemon to be ready
    local max_attempts=30
    local attempt=0

    while ! docker ps &>/dev/null; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            error_exit "Docker daemon failed to start"
        fi
        sleep 1
    done

    success "Docker service enabled and started"
}

# Setup Docker daemon configuration
setup_docker_daemon() {
    log "INFO" "Configuring Docker daemon..."

    mkdir -p /etc/docker

    cat > /etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "10"
  },
  "storage-driver": "overlay2",
  "insecure-registries": [],
  "live-restore": true,
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 5,
  "bridge": "none"
}
EOF

    systemctl restart docker

    # Wait for Docker daemon to be ready
    sleep 5

    success "Docker daemon configured"
}

# Setup user permissions
setup_user_permissions() {
    log "INFO" "Setting up Docker user permissions..."

    # Add docker group if it doesn't exist
    if ! getent group docker > /dev/null; then
        groupadd docker
    fi

    # Add current user to docker group if not root
    if [[ "$SUDO_USER" ]]; then
        usermod -aG docker "$SUDO_USER"
        info "User $SUDO_USER added to docker group (log out and back in for changes to take effect)"
    fi

    success "Docker user permissions configured"
}

# Install Docker Compose
install_docker_compose() {
    log "INFO" "Installing Docker Compose..."

    # Check if docker compose (V2) is available
    if docker compose version &>/dev/null; then
        success "Docker Compose (V2) already available"
        return 0
    fi

    # Install Docker Compose V2 plugin
    mkdir -p /usr/local/lib/docker/cli-plugins

    if [[ "$DOCKER_COMPOSE_VERSION" == "latest" ]]; then
        local compose_version
        compose_version=$(curl -fsSL https://api.github.com/repos/docker/compose/releases/latest | grep -Po '"tag_name": "\K[^"]*')
    else
        local compose_version="$DOCKER_COMPOSE_VERSION"
    fi

    local arch
    arch=$(uname -m)

    case "$arch" in
        x86_64)
            arch="x86_64"
            ;;
        aarch64)
            arch="aarch64"
            ;;
        *)
            error_exit "Unsupported architecture: $arch"
            ;;
    esac

    local compose_url="https://github.com/docker/compose/releases/download/${compose_version}/docker-compose-$(uname -s)-${arch}"

    curl -fsSL "$compose_url" -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

    success "Docker Compose installed (version: $compose_version)"
}

# Verify Docker Compose installation
verify_docker_compose_installation() {
    log "INFO" "Verifying Docker Compose installation..."

    docker compose version || error_exit "Docker Compose installation verification failed"

    success "Docker Compose installation verified"
}

# Setup project Docker configuration
setup_project_docker_config() {
    log "INFO" "Setting up project Docker configuration..."

    mkdir -p "$PROJECT_ROOT"/{logs,data}

    success "Project Docker configuration set up"
}

# Run final checks
run_final_checks() {
    log "INFO" "Running final checks..."

    info "Docker environment information:"
    echo "  Docker Version: $(docker --version)"
    echo "  Docker Compose Version: $(docker compose version --short)"
    echo "  Docker Status: $(systemctl is-active docker)"
    echo "  Docker Storage Driver: $(docker info --format '{{.Driver}}')"
    echo "  Docker Rootless Mode: $(docker info --format '{{.SecurityOptions}}')"

    success "Docker setup completed successfully!"
}

# Main execution
main() {
    log "INFO" "=== Docker Installation Started ==="

    check_root

    local distro
    distro=$(detect_distro)

    if [[ ! "$distro" =~ ^(debian|ubuntu)$ ]]; then
        warning "This script is optimized for Debian/Ubuntu. Detected: $distro"
    fi

    log "INFO" "Detected distribution: $distro"

    uninstall_old_docker
    add_docker_gpg_key
    setup_docker_repository
    install_docker
    verify_docker_installation
    enable_docker_service
    setup_docker_daemon
    setup_user_permissions
    install_docker_compose
    verify_docker_compose_installation
    setup_project_docker_config
    run_final_checks

    log "INFO" "=== Docker Installation Completed ==="
}

# Run main function
main "$@"
