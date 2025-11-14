"""
Quantum Trader AI - IP Whitelist Management
Production-grade IP access control with geo-blocking

CRITICAL CONSTRAINTS:
- All numeric values use Decimal, NEVER float
- All data operations use polars DataFrame
- All external calls wrapped in try/except with retry logic
- Complete type hints everywhere
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import yaml
import polars as pl
import ipaddress
import re

from quantum_trader.models import AuditLog
from quantum_trader.database.timeseries import TimeSeriesDB


logger = logging.getLogger(__name__)


class IPWhitelistManager:
    """
    IP whitelist management system

    Features:
    - Dynamic whitelist updates
    - CIDR range support
    - Geo-blocking
    - IP reputation checking
    - Whitelist violation alerts
    """

    def __init__(
        self,
        config_path: Path = Path("/home/user/FritzellSama/config/environments/production.yaml")
    ) -> None:
        """Initialize IP whitelist manager with configuration"""
        self.config = self._load_config(config_path)

        # Security configuration
        security_config = self.config.get("security", {})
        self.tls_enabled = security_config.get("tls_enabled", True)

        # IP whitelist (stored as CIDR ranges for efficiency)
        self._whitelist: Set[ipaddress.IPv4Network | ipaddress.IPv6Network] = set()

        # Blacklist (blocked IPs)
        self._blacklist: Set[ipaddress.IPv4Network | ipaddress.IPv6Network] = set()

        # IP metadata (geolocation, reputation, etc.)
        self._ip_metadata: Dict[str, Dict] = {}

        # Blocked countries (ISO country codes)
        self._blocked_countries: Set[str] = {
            # Example: Add countries based on regulatory requirements
            # "KP", "IR", "SY"  # North Korea, Iran, Syria
        }

        # Allowed countries (if set, only these are allowed)
        self._allowed_countries: Optional[Set[str]] = None

        # Rate limiting per IP
        self._ip_access_log: Dict[str, List[datetime]] = {}

        # Suspicious activity tracking
        self._suspicious_ips: Dict[str, Dict] = {}

        # Database connection
        self.db: Optional[TimeSeriesDB] = None

        # Retry configuration
        self.retry_attempts = 3
        self.retry_delay_ms = 1000

        logger.info("IPWhitelistManager initialized")

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    async def initialize(self) -> None:
        """Initialize database connections and load whitelist"""
        try:
            self.db = TimeSeriesDB()
            await self.db.connect()

            # Load whitelist from database
            await self._load_whitelist()

            # Load blacklist
            await self._load_blacklist()

            logger.info("IPWhitelistManager initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize IPWhitelistManager: {e}")
            raise

    async def _load_whitelist(self) -> None:
        """Load IP whitelist from database"""
        if not self.db:
            return

        try:
            whitelist_df = await self.db.query_ip_whitelist()

            for row in whitelist_df.iter_rows(named=True):
                ip_range = row["ip_range"]
                try:
                    network = ipaddress.ip_network(ip_range, strict=False)
                    self._whitelist.add(network)
                except ValueError as e:
                    logger.error(f"Invalid IP range in whitelist: {ip_range}: {e}")

            logger.info(f"Loaded {len(self._whitelist)} IP ranges to whitelist")

        except Exception as e:
            logger.error(f"Failed to load IP whitelist: {e}")

    async def _load_blacklist(self) -> None:
        """Load IP blacklist from database"""
        if not self.db:
            return

        try:
            blacklist_df = await self.db.query_ip_blacklist()

            for row in blacklist_df.iter_rows(named=True):
                ip_range = row["ip_range"]
                try:
                    network = ipaddress.ip_network(ip_range, strict=False)
                    self._blacklist.add(network)
                except ValueError as e:
                    logger.error(f"Invalid IP range in blacklist: {ip_range}: {e}")

            logger.info(f"Loaded {len(self._blacklist)} IP ranges to blacklist")

        except Exception as e:
            logger.error(f"Failed to load IP blacklist: {e}")

    async def add_to_whitelist(
        self,
        ip_range: str,
        added_by: str,
        description: Optional[str] = None
    ) -> bool:
        """
        Add IP or CIDR range to whitelist

        Args:
            ip_range: IP address or CIDR range (e.g., "192.168.1.0/24")
            added_by: User adding the IP
            description: Optional description

        Returns:
            True if successful
        """
        try:
            # Validate IP range
            network = ipaddress.ip_network(ip_range, strict=False)

            # Add to whitelist
            self._whitelist.add(network)

            # Persist to database
            if self.db:
                await self.db.insert_ip_whitelist(
                    ip_range=str(network),
                    added_by=added_by,
                    description=description or "No description"
                )

            # Audit log
            await self._log_security_event(
                "IP_WHITELIST_ADD",
                added_by,
                {
                    "ip_range": str(network),
                    "description": description,
                    "result": "SUCCESS"
                }
            )

            logger.info(f"Added {network} to whitelist by {added_by}")
            return True

        except ValueError as e:
            logger.error(f"Invalid IP range: {ip_range}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error adding to whitelist: {e}")
            return False

    async def remove_from_whitelist(
        self,
        ip_range: str,
        removed_by: str
    ) -> bool:
        """
        Remove IP or CIDR range from whitelist

        Args:
            ip_range: IP address or CIDR range
            removed_by: User removing the IP

        Returns:
            True if successful
        """
        try:
            network = ipaddress.ip_network(ip_range, strict=False)

            if network in self._whitelist:
                self._whitelist.remove(network)

                # Persist to database
                if self.db:
                    await self.db.delete_ip_whitelist(str(network))

                # Audit log
                await self._log_security_event(
                    "IP_WHITELIST_REMOVE",
                    removed_by,
                    {
                        "ip_range": str(network),
                        "result": "SUCCESS"
                    }
                )

                logger.info(f"Removed {network} from whitelist by {removed_by}")
                return True
            else:
                logger.warning(f"IP range {network} not in whitelist")
                return False

        except ValueError as e:
            logger.error(f"Invalid IP range: {ip_range}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error removing from whitelist: {e}")
            return False

    async def add_to_blacklist(
        self,
        ip_range: str,
        added_by: str,
        reason: str
    ) -> bool:
        """
        Add IP or CIDR range to blacklist

        Args:
            ip_range: IP address or CIDR range
            added_by: User adding the IP
            reason: Reason for blacklisting

        Returns:
            True if successful
        """
        try:
            network = ipaddress.ip_network(ip_range, strict=False)

            # Add to blacklist
            self._blacklist.add(network)

            # Remove from whitelist if present
            if network in self._whitelist:
                self._whitelist.remove(network)

            # Persist to database
            if self.db:
                await self.db.insert_ip_blacklist(
                    ip_range=str(network),
                    added_by=added_by,
                    reason=reason
                )

            # Audit log
            await self._log_security_event(
                "IP_BLACKLIST_ADD",
                added_by,
                {
                    "ip_range": str(network),
                    "reason": reason,
                    "result": "SUCCESS",
                    "severity": "WARNING"
                }
            )

            logger.warning(f"Added {network} to blacklist by {added_by}: {reason}")
            return True

        except ValueError as e:
            logger.error(f"Invalid IP range: {ip_range}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error adding to blacklist: {e}")
            return False

    async def check_ip_allowed(
        self,
        ip_address: str,
        user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if IP address is allowed

        Args:
            ip_address: IP address to check
            user_id: Optional user identifier for logging

        Returns:
            Tuple of (is_allowed, reason)
        """
        try:
            ip_obj = ipaddress.ip_address(ip_address)

            # Check blacklist first (highest priority)
            if self._is_ip_in_list(ip_obj, self._blacklist):
                reason = "IP is blacklisted"
                await self._log_access_attempt(ip_address, user_id, False, reason)
                return False, reason

            # Check whitelist
            if len(self._whitelist) > 0:
                if not self._is_ip_in_list(ip_obj, self._whitelist):
                    reason = "IP not in whitelist"
                    await self._log_access_attempt(ip_address, user_id, False, reason)
                    await self._track_suspicious_activity(ip_address, "NOT_WHITELISTED")
                    return False, reason

            # Check geo-blocking (if configured)
            geo_allowed, geo_reason = await self._check_geo_location(ip_address)
            if not geo_allowed:
                await self._log_access_attempt(ip_address, user_id, False, geo_reason)
                return False, geo_reason

            # Check IP reputation
            reputation_ok, reputation_reason = await self._check_ip_reputation(ip_address)
            if not reputation_ok:
                await self._log_access_attempt(ip_address, user_id, False, reputation_reason)
                return False, reputation_reason

            # Check suspicious activity
            if self._is_suspicious_ip(ip_address):
                reason = "Suspicious activity detected"
                await self._log_access_attempt(ip_address, user_id, False, reason)
                return False, reason

            # IP is allowed
            await self._log_access_attempt(ip_address, user_id, True, "Access granted")
            return True, None

        except ValueError as e:
            logger.error(f"Invalid IP address: {ip_address}: {e}")
            return False, "Invalid IP address"
        except Exception as e:
            logger.error(f"Error checking IP: {e}")
            return False, "Error checking IP"

    def _is_ip_in_list(
        self,
        ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address,
        ip_list: Set[ipaddress.IPv4Network | ipaddress.IPv6Network]
    ) -> bool:
        """Check if IP is in a list of networks"""
        for network in ip_list:
            if ip_obj in network:
                return True
        return False

    async def _check_geo_location(
        self,
        ip_address: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if IP geo-location is allowed

        Args:
            ip_address: IP address to check

        Returns:
            Tuple of (is_allowed, reason)
        """
        try:
            # Get country for IP (in production, use MaxMind GeoIP2 or similar)
            country_code = await self._get_country_for_ip(ip_address)

            if country_code is None:
                # Cannot determine country - allow by default
                return True, None

            # Check if country is blocked
            if country_code in self._blocked_countries:
                return False, f"Country {country_code} is blocked"

            # Check if only specific countries are allowed
            if self._allowed_countries is not None:
                if country_code not in self._allowed_countries:
                    return False, f"Country {country_code} not in allowed list"

            return True, None

        except Exception as e:
            logger.error(f"Error checking geo-location: {e}")
            return True, None  # Allow on error to avoid blocking legitimate traffic

    async def _get_country_for_ip(self, ip_address: str) -> Optional[str]:
        """
        Get country code for IP address

        In production, integrate with GeoIP2 database or API.
        This is a placeholder.
        """
        # Placeholder - return None
        # In production:
        # import geoip2.database
        # reader = geoip2.database.Reader('GeoLite2-Country.mmdb')
        # response = reader.country(ip_address)
        # return response.country.iso_code

        return None

    async def _check_ip_reputation(
        self,
        ip_address: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check IP reputation against known threat databases

        Args:
            ip_address: IP address to check

        Returns:
            Tuple of (is_ok, reason)
        """
        try:
            # Check cached metadata
            if ip_address in self._ip_metadata:
                metadata = self._ip_metadata[ip_address]
                reputation = metadata.get("reputation", "unknown")

                if reputation == "malicious":
                    return False, "IP has malicious reputation"

            # In production, integrate with threat intelligence services:
            # - AbuseIPDB
            # - IPQualityScore
            # - VirusTotal
            # etc.

            return True, None

        except Exception as e:
            logger.error(f"Error checking IP reputation: {e}")
            return True, None  # Allow on error

    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP has shown suspicious activity"""
        if ip_address in self._suspicious_ips:
            suspicious_data = self._suspicious_ips[ip_address]

            # Check if too many violations
            violation_count = suspicious_data.get("violation_count", 0)
            if violation_count >= 5:
                return True

            # Check if recent violations
            last_violation = suspicious_data.get("last_violation")
            if last_violation:
                time_since = datetime.utcnow() - last_violation
                if time_since < timedelta(hours=1) and violation_count >= 3:
                    return True

        return False

    async def _track_suspicious_activity(
        self,
        ip_address: str,
        activity_type: str
    ) -> None:
        """Track suspicious activity from IP"""
        if ip_address not in self._suspicious_ips:
            self._suspicious_ips[ip_address] = {
                "violation_count": 0,
                "first_seen": datetime.utcnow(),
                "activities": []
            }

        suspicious_data = self._suspicious_ips[ip_address]
        suspicious_data["violation_count"] += 1
        suspicious_data["last_violation"] = datetime.utcnow()
        suspicious_data["activities"].append({
            "type": activity_type,
            "timestamp": datetime.utcnow()
        })

        # Auto-blacklist if too many violations
        if suspicious_data["violation_count"] >= 10:
            await self.add_to_blacklist(
                ip_address,
                "system",
                f"Auto-blacklisted after {suspicious_data['violation_count']} violations"
            )

    async def _log_access_attempt(
        self,
        ip_address: str,
        user_id: Optional[str],
        allowed: bool,
        reason: str
    ) -> None:
        """Log IP access attempt"""
        # Track in memory
        if ip_address not in self._ip_access_log:
            self._ip_access_log[ip_address] = []

        self._ip_access_log[ip_address].append(datetime.utcnow())

        # Clean old entries (keep last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self._ip_access_log[ip_address] = [
            ts for ts in self._ip_access_log[ip_address]
            if ts > cutoff
        ]

        # Audit log for denied access
        if not allowed:
            await self._log_security_event(
                "IP_ACCESS_DENIED",
                user_id or "unknown",
                {
                    "ip_address": ip_address,
                    "reason": reason,
                    "severity": "WARNING"
                }
            )

    async def _log_security_event(
        self,
        operation: str,
        user_id: str,
        details: Dict
    ) -> None:
        """Log security event to audit trail"""
        try:
            severity = details.get("severity", "INFO")

            audit_log = AuditLog(
                timestamp=datetime.utcnow(),
                operation=operation,
                user_id=user_id,
                component="IPWhitelistManager",
                severity=severity,
                details=details,
                ip_address=details.get("ip_address"),
                result=details.get("result", "SUCCESS")
            )

            if self.db:
                await self.db.insert_audit_log(audit_log)

        except Exception as e:
            logger.error(f"Failed to log security event: {e}")

    async def get_whitelist(self) -> List[str]:
        """Get current whitelist"""
        return [str(network) for network in self._whitelist]

    async def get_blacklist(self) -> List[str]:
        """Get current blacklist"""
        return [str(network) for network in self._blacklist]

    async def get_ip_access_stats(self, ip_address: str) -> Dict:
        """
        Get access statistics for an IP

        Args:
            ip_address: IP address

        Returns:
            Statistics dictionary
        """
        try:
            access_count = len(self._ip_access_log.get(ip_address, []))

            suspicious_data = self._suspicious_ips.get(ip_address, {})

            is_whitelisted = self._is_ip_in_list(
                ipaddress.ip_address(ip_address),
                self._whitelist
            )

            is_blacklisted = self._is_ip_in_list(
                ipaddress.ip_address(ip_address),
                self._blacklist
            )

            stats = {
                "ip_address": ip_address,
                "access_count_24h": access_count,
                "is_whitelisted": is_whitelisted,
                "is_blacklisted": is_blacklisted,
                "is_suspicious": self._is_suspicious_ip(ip_address),
                "violation_count": suspicious_data.get("violation_count", 0),
                "last_violation": suspicious_data.get("last_violation").isoformat() if suspicious_data.get("last_violation") else None
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting IP stats: {e}")
            return {"error": str(e)}

    async def get_whitelist_report(self) -> Dict:
        """
        Generate comprehensive whitelist report

        Returns:
            Report dictionary
        """
        try:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "whitelist_count": len(self._whitelist),
                "blacklist_count": len(self._blacklist),
                "suspicious_ips": len(self._suspicious_ips),
                "whitelist_entries": [str(network) for network in self._whitelist],
                "blacklist_entries": [str(network) for network in self._blacklist],
                "top_suspicious_ips": [
                    {
                        "ip": ip,
                        "violations": data["violation_count"],
                        "last_violation": data["last_violation"].isoformat() if data.get("last_violation") else None
                    }
                    for ip, data in sorted(
                        self._suspicious_ips.items(),
                        key=lambda x: x[1]["violation_count"],
                        reverse=True
                    )[:10]
                ]
            }

            return report

        except Exception as e:
            logger.error(f"Error generating whitelist report: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.db:
            await self.db.disconnect()

        logger.info("IPWhitelistManager cleanup complete")
