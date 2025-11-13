"""
IP Whitelist Module - Network Access Control
CRITICAL: Production-ready IP whitelisting with CIDR support
"""

import logging
import ipaddress
from typing import Set, List, Optional
from datetime import datetime
import asyncio

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class IPWhitelist:
    """Production IP whitelisting with CIDR range support"""

    def __init__(self):
        """Initialize IP whitelist from configuration"""
        self._config = get_config()
        self._allowed_ips: Set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        self._allowed_networks: Set[ipaddress.IPv4Network | ipaddress.IPv6Network] = set()
        self._enabled: bool = True
        self._strict_mode: bool = True
        self._lock = asyncio.Lock()

        # Load configuration
        self._load_config()

        logger.info(
            f"IP whitelist initialized: {len(self._allowed_ips)} IPs, "
            f"{len(self._allowed_networks)} CIDR ranges, enabled={self._enabled}"
        )

    def _load_config(self) -> None:
        """Load IP whitelist configuration"""
        try:
            # Load whitelist settings
            self._enabled = self._config.get_bool('security', 'ip_whitelist.enabled', True)
            self._strict_mode = self._config.get_bool('security', 'ip_whitelist.strict_mode', True)

            # Load allowed IPs from config
            allowed_ips_str = self._config.get('security', 'ip_whitelist.allowed_ips', '')
            if allowed_ips_str:
                ip_list = [ip.strip() for ip in allowed_ips_str.split(',') if ip.strip()]
                for ip_str in ip_list:
                    try:
                        ip_addr = ipaddress.ip_address(ip_str)
                        self._allowed_ips.add(ip_addr)
                    except ValueError as e:
                        logger.error(f"Invalid IP address in config: {ip_str}: {e}")

            # Load allowed CIDR ranges from config
            allowed_cidr_str = self._config.get('security', 'ip_whitelist.allowed_cidr_ranges', '')
            if allowed_cidr_str:
                cidr_list = [cidr.strip() for cidr in allowed_cidr_str.split(',') if cidr.strip()]
                for cidr_str in cidr_list:
                    try:
                        network = ipaddress.ip_network(cidr_str, strict=False)
                        self._allowed_networks.add(network)
                    except ValueError as e:
                        logger.error(f"Invalid CIDR range in config: {cidr_str}: {e}")

            logger.info("IP whitelist configuration loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load IP whitelist configuration: {e}")
            # Set safe defaults
            self._enabled = True
            self._strict_mode = True

    async def load_whitelist(self) -> bool:
        """
        Reload whitelist from configuration

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                # Clear existing entries
                self._allowed_ips.clear()
                self._allowed_networks.clear()

                # Reload configuration
                self._config.reload('security')
                self._load_config()

                logger.info("IP whitelist reloaded from configuration")
                return True

            except Exception as e:
                logger.error(f"Failed to reload IP whitelist: {e}")
                return False

    async def check_ip(self, ip_address: str) -> bool:
        """
        Check if an IP address is whitelisted

        Args:
            ip_address: IP address string to check

        Returns:
            True if allowed, False otherwise
        """
        try:
            # If whitelisting is disabled, allow all
            if not self._enabled:
                return True

            # Parse IP address
            try:
                ip_obj = ipaddress.ip_address(ip_address)
            except ValueError as e:
                logger.error(f"Invalid IP address format: {ip_address}: {e}")
                return False

            # Check direct IP match
            if ip_obj in self._allowed_ips:
                logger.debug(f"IP {ip_address} allowed (direct match)")
                return True

            # Check CIDR range match
            for network in self._allowed_networks:
                if ip_obj in network:
                    logger.debug(f"IP {ip_address} allowed (CIDR match: {network})")
                    return True

            # IP not found in whitelist
            if self._strict_mode:
                logger.warning(f"IP {ip_address} BLOCKED by whitelist (strict mode)")
            else:
                logger.info(f"IP {ip_address} not in whitelist (non-strict mode)")

            return not self._strict_mode

        except Exception as e:
            logger.error(f"Error checking IP {ip_address}: {e}")
            # Fail secure - deny access on error
            return False

    async def add_ip(self, ip_address: str) -> bool:
        """
        Add an IP address to the whitelist

        Args:
            ip_address: IP address string to add

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                # Parse and validate IP address
                try:
                    ip_obj = ipaddress.ip_address(ip_address)
                except ValueError as e:
                    logger.error(f"Invalid IP address format: {ip_address}: {e}")
                    return False

                # Check if already exists
                if ip_obj in self._allowed_ips:
                    logger.info(f"IP {ip_address} already in whitelist")
                    return True

                # Add to whitelist
                self._allowed_ips.add(ip_obj)

                logger.info(f"Added IP {ip_address} to whitelist")
                return True

            except Exception as e:
                logger.error(f"Failed to add IP {ip_address} to whitelist: {e}")
                return False

    async def remove_ip(self, ip_address: str) -> bool:
        """
        Remove an IP address from the whitelist

        Args:
            ip_address: IP address string to remove

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                # Parse IP address
                try:
                    ip_obj = ipaddress.ip_address(ip_address)
                except ValueError as e:
                    logger.error(f"Invalid IP address format: {ip_address}: {e}")
                    return False

                # Remove from whitelist
                if ip_obj in self._allowed_ips:
                    self._allowed_ips.remove(ip_obj)
                    logger.info(f"Removed IP {ip_address} from whitelist")
                    return True
                else:
                    logger.warning(f"IP {ip_address} not found in whitelist")
                    return False

            except Exception as e:
                logger.error(f"Failed to remove IP {ip_address} from whitelist: {e}")
                return False

    async def add_cidr(self, cidr_range: str) -> bool:
        """
        Add a CIDR range to the whitelist

        Args:
            cidr_range: CIDR range string (e.g., "192.168.1.0/24")

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                # Parse and validate CIDR range
                try:
                    network = ipaddress.ip_network(cidr_range, strict=False)
                except ValueError as e:
                    logger.error(f"Invalid CIDR range format: {cidr_range}: {e}")
                    return False

                # Check if already exists
                if network in self._allowed_networks:
                    logger.info(f"CIDR range {cidr_range} already in whitelist")
                    return True

                # Add to whitelist
                self._allowed_networks.add(network)

                logger.info(f"Added CIDR range {cidr_range} to whitelist")
                return True

            except Exception as e:
                logger.error(f"Failed to add CIDR range {cidr_range} to whitelist: {e}")
                return False

    async def remove_cidr(self, cidr_range: str) -> bool:
        """
        Remove a CIDR range from the whitelist

        Args:
            cidr_range: CIDR range string to remove

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                # Parse CIDR range
                try:
                    network = ipaddress.ip_network(cidr_range, strict=False)
                except ValueError as e:
                    logger.error(f"Invalid CIDR range format: {cidr_range}: {e}")
                    return False

                # Remove from whitelist
                if network in self._allowed_networks:
                    self._allowed_networks.remove(network)
                    logger.info(f"Removed CIDR range {cidr_range} from whitelist")
                    return True
                else:
                    logger.warning(f"CIDR range {cidr_range} not found in whitelist")
                    return False

            except Exception as e:
                logger.error(f"Failed to remove CIDR range {cidr_range} from whitelist: {e}")
                return False

    async def check_cidr(self, ip_address: str, cidr_range: str) -> bool:
        """
        Check if an IP address is within a specific CIDR range

        Args:
            ip_address: IP address string to check
            cidr_range: CIDR range string to check against

        Returns:
            True if IP is in CIDR range, False otherwise
        """
        try:
            # Parse IP address
            try:
                ip_obj = ipaddress.ip_address(ip_address)
            except ValueError as e:
                logger.error(f"Invalid IP address format: {ip_address}: {e}")
                return False

            # Parse CIDR range
            try:
                network = ipaddress.ip_network(cidr_range, strict=False)
            except ValueError as e:
                logger.error(f"Invalid CIDR range format: {cidr_range}: {e}")
                return False

            # Check if IP is in network
            result = ip_obj in network

            if result:
                logger.debug(f"IP {ip_address} is in CIDR range {cidr_range}")
            else:
                logger.debug(f"IP {ip_address} is NOT in CIDR range {cidr_range}")

            return result

        except Exception as e:
            logger.error(f"Error checking IP {ip_address} against CIDR {cidr_range}: {e}")
            return False

    async def get_whitelisted_ips(self) -> List[str]:
        """
        Get all whitelisted IP addresses

        Returns:
            List of IP address strings
        """
        try:
            return [str(ip) for ip in self._allowed_ips]
        except Exception as e:
            logger.error(f"Failed to get whitelisted IPs: {e}")
            return []

    async def get_whitelisted_cidrs(self) -> List[str]:
        """
        Get all whitelisted CIDR ranges

        Returns:
            List of CIDR range strings
        """
        try:
            return [str(network) for network in self._allowed_networks]
        except Exception as e:
            logger.error(f"Failed to get whitelisted CIDR ranges: {e}")
            return []

    async def is_enabled(self) -> bool:
        """
        Check if IP whitelisting is enabled

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled

    async def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable IP whitelisting

        Args:
            enabled: True to enable, False to disable
        """
        async with self._lock:
            self._enabled = enabled
            logger.info(f"IP whitelisting {'enabled' if enabled else 'disabled'}")

    async def is_strict_mode(self) -> bool:
        """
        Check if strict mode is enabled

        Returns:
            True if strict mode enabled, False otherwise
        """
        return self._strict_mode

    async def set_strict_mode(self, strict: bool) -> None:
        """
        Enable or disable strict mode

        Args:
            strict: True for strict mode, False for permissive mode
        """
        async with self._lock:
            self._strict_mode = strict
            logger.info(f"IP whitelist strict mode {'enabled' if strict else 'disabled'}")

    async def clear_whitelist(self) -> bool:
        """
        Clear all whitelisted IPs and CIDR ranges

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                ip_count = len(self._allowed_ips)
                cidr_count = len(self._allowed_networks)

                self._allowed_ips.clear()
                self._allowed_networks.clear()

                logger.warning(
                    f"Cleared IP whitelist: removed {ip_count} IPs and {cidr_count} CIDR ranges"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to clear whitelist: {e}")
                return False

    async def get_whitelist_size(self) -> tuple[int, int]:
        """
        Get whitelist size (IP count, CIDR count)

        Returns:
            Tuple of (ip_count, cidr_count)
        """
        try:
            return len(self._allowed_ips), len(self._allowed_networks)
        except Exception as e:
            logger.error(f"Failed to get whitelist size: {e}")
            return 0, 0

    async def validate_ip_list(self, ip_list: List[str]) -> tuple[List[str], List[str]]:
        """
        Validate a list of IP addresses

        Args:
            ip_list: List of IP address strings

        Returns:
            Tuple of (valid_ips, invalid_ips)
        """
        valid_ips = []
        invalid_ips = []

        for ip_str in ip_list:
            try:
                ipaddress.ip_address(ip_str)
                valid_ips.append(ip_str)
            except ValueError:
                invalid_ips.append(ip_str)

        return valid_ips, invalid_ips

    async def validate_cidr_list(self, cidr_list: List[str]) -> tuple[List[str], List[str]]:
        """
        Validate a list of CIDR ranges

        Args:
            cidr_list: List of CIDR range strings

        Returns:
            Tuple of (valid_cidrs, invalid_cidrs)
        """
        valid_cidrs = []
        invalid_cidrs = []

        for cidr_str in cidr_list:
            try:
                ipaddress.ip_network(cidr_str, strict=False)
                valid_cidrs.append(cidr_str)
            except ValueError:
                invalid_cidrs.append(cidr_str)

        return valid_cidrs, invalid_cidrs
