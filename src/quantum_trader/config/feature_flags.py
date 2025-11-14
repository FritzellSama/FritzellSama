"""
Feature Flags - Dynamic feature toggling and A/B testing.

This module provides runtime feature flag management for gradual rollouts,
A/B testing, and emergency feature disabling.
"""

import asyncio
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from structlog import get_logger

logger = get_logger(__name__)


class FlagStatus(Enum):
    """Feature flag status types."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    PERCENTAGE = "percentage"
    CONDITIONAL = "conditional"


@dataclass
class FeatureFlag:
    """
    Individual feature flag configuration.

    Attributes:
        name: Unique flag name
        status: Current flag status
        description: Human-readable description
        percentage: Percentage of users/requests to enable (0-100)
        conditions: Conditional evaluation rules
        enabled_for: Specific identifiers with flag enabled
        disabled_for: Specific identifiers with flag disabled
        metadata: Additional flag metadata
    """

    name: str
    status: FlagStatus
    description: str
    percentage: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled_for: Set[str] = field(default_factory=set)
    disabled_for: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate flag configuration."""
        if not (0 <= self.percentage <= 100):
            raise ValueError(f"Percentage must be 0-100, got {self.percentage}")


class FeatureFlagManager:
    """
    Manages feature flags for runtime feature toggling.

    Supports various flag types including on/off, percentage rollouts,
    and conditional flags based on context.

    Attributes:
        flags: Dictionary of all feature flags
        default_value: Default value when flag not found

    Example:
        >>> flag_mgr = FeatureFlagManager(config)
        >>> if await flag_mgr.is_enabled('new_ml_model', user_id='user123'):
        ...     # Use new ML model
        >>> else:
        ...     # Use legacy model
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize feature flag manager.

        Args:
            config: Configuration dict with keys:
                - flags: Dictionary of flag configurations
                - default_enabled: Default value for unknown flags
                - refresh_interval: Interval to refresh flags from remote

        Raises:
            ValueError: If config invalid
        """
        self.config = config
        self._validate_config()

        self.flags: Dict[str, FeatureFlag] = {}
        self.default_value = bool(config.get('default_enabled', False))
        self.refresh_interval = int(config.get('refresh_interval', 60))

        # Load initial flags
        self._load_flags(config.get('flags', {}))

        # Evaluation counters for analytics
        self._evaluation_counts: Dict[str, int] = {}
        self._enabled_counts: Dict[str, int] = {}

        logger.info(
            "FeatureFlagManager initialized",
            flags_count=len(self.flags),
            default_enabled=self.default_value
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if 'flags' not in self.config:
            logger.warning("No flags defined in config")

    def _load_flags(self, flags_config: Dict[str, Any]) -> None:
        """
        Load flags from configuration.

        Args:
            flags_config: Dictionary of flag configurations
        """
        for flag_name, flag_data in flags_config.items():
            try:
                status_str = flag_data.get('status', 'disabled')
                status = FlagStatus(status_str)

                flag = FeatureFlag(
                    name=flag_name,
                    status=status,
                    description=flag_data.get('description', ''),
                    percentage=int(flag_data.get('percentage', 0)),
                    conditions=flag_data.get('conditions', {}),
                    enabled_for=set(flag_data.get('enabled_for', [])),
                    disabled_for=set(flag_data.get('disabled_for', [])),
                    metadata=flag_data.get('metadata', {})
                )

                self.flags[flag_name] = flag

                logger.debug(
                    "Loaded feature flag",
                    name=flag_name,
                    status=status.value
                )

            except Exception as e:
                logger.error(
                    "Failed to load feature flag",
                    name=flag_name,
                    error=str(e)
                )

    async def is_enabled(
        self,
        flag_name: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> bool:
        """
        Check if feature flag is enabled.

        Args:
            flag_name: Name of the feature flag
            context: Additional context for conditional evaluation
            user_id: User identifier for percentage/targeting
            strategy_id: Strategy identifier for percentage/targeting

        Returns:
            True if feature is enabled, False otherwise

        Example:
            >>> enabled = await flag_mgr.is_enabled(
            ...     'advanced_risk_model',
            ...     context={'exchange': 'binance'},
            ...     strategy_id='momentum_v2'
            ... )
        """
        try:
            # Track evaluation
            self._evaluation_counts[flag_name] = self._evaluation_counts.get(flag_name, 0) + 1

            # Check if flag exists
            if flag_name not in self.flags:
                logger.debug(
                    "Unknown feature flag, using default",
                    flag=flag_name,
                    default=self.default_value
                )
                return self.default_value

            flag = self.flags[flag_name]

            # Build identifier for targeting
            identifier = user_id or strategy_id or ''

            # Check explicit disabled list first
            if identifier and identifier in flag.disabled_for:
                return False

            # Check explicit enabled list
            if identifier and identifier in flag.enabled_for:
                self._enabled_counts[flag_name] = self._enabled_counts.get(flag_name, 0) + 1
                return True

            # Evaluate based on status
            if flag.status == FlagStatus.ENABLED:
                self._enabled_counts[flag_name] = self._enabled_counts.get(flag_name, 0) + 1
                return True

            elif flag.status == FlagStatus.DISABLED:
                return False

            elif flag.status == FlagStatus.PERCENTAGE:
                enabled = self._evaluate_percentage(flag, identifier)
                if enabled:
                    self._enabled_counts[flag_name] = self._enabled_counts.get(flag_name, 0) + 1
                return enabled

            elif flag.status == FlagStatus.CONDITIONAL:
                enabled = self._evaluate_conditions(flag, context or {})
                if enabled:
                    self._enabled_counts[flag_name] = self._enabled_counts.get(flag_name, 0) + 1
                return enabled

            return self.default_value

        except Exception as e:
            logger.error(
                "Feature flag evaluation failed",
                flag=flag_name,
                error=str(e)
            )
            return self.default_value

    def _evaluate_percentage(self, flag: FeatureFlag, identifier: str) -> bool:
        """
        Evaluate percentage-based flag.

        Uses consistent hashing to ensure same identifier always gets same result.

        Args:
            flag: Feature flag
            identifier: User/strategy identifier

        Returns:
            True if within percentage threshold
        """
        if not identifier:
            # Random percentage without identifier
            import random
            return random.randint(0, 100) < flag.percentage

        # Consistent hashing based on identifier
        hash_value = hash(f"{flag.name}:{identifier}")
        bucket = abs(hash_value) % 100

        return bucket < flag.percentage

    def _evaluate_conditions(
        self,
        flag: FeatureFlag,
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate conditional flag based on context.

        Args:
            flag: Feature flag with conditions
            context: Evaluation context

        Returns:
            True if all conditions met
        """
        if not flag.conditions:
            return False

        for key, expected_value in flag.conditions.items():
            actual_value = context.get(key)

            # Handle list of acceptable values
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            # Handle single value
            elif actual_value != expected_value:
                return False

        return True

    def enable_flag(self, flag_name: str) -> None:
        """
        Enable a feature flag.

        Args:
            flag_name: Name of flag to enable
        """
        if flag_name in self.flags:
            self.flags[flag_name].status = FlagStatus.ENABLED
            logger.info("Feature flag enabled", flag=flag_name)
        else:
            logger.warning("Attempted to enable unknown flag", flag=flag_name)

    def disable_flag(self, flag_name: str) -> None:
        """
        Disable a feature flag.

        Args:
            flag_name: Name of flag to disable
        """
        if flag_name in self.flags:
            self.flags[flag_name].status = FlagStatus.DISABLED
            logger.info("Feature flag disabled", flag=flag_name)
        else:
            logger.warning("Attempted to disable unknown flag", flag=flag_name)

    def set_percentage(self, flag_name: str, percentage: int) -> None:
        """
        Set percentage rollout for a flag.

        Args:
            flag_name: Name of flag
            percentage: Percentage (0-100)

        Raises:
            ValueError: If percentage invalid
        """
        if not (0 <= percentage <= 100):
            raise ValueError(f"Percentage must be 0-100, got {percentage}")

        if flag_name in self.flags:
            self.flags[flag_name].status = FlagStatus.PERCENTAGE
            self.flags[flag_name].percentage = percentage
            logger.info(
                "Feature flag percentage updated",
                flag=flag_name,
                percentage=percentage
            )
        else:
            logger.warning("Attempted to set percentage for unknown flag", flag=flag_name)

    def add_flag(self, flag: FeatureFlag) -> None:
        """
        Add a new feature flag.

        Args:
            flag: FeatureFlag to add
        """
        self.flags[flag.name] = flag
        logger.info("Feature flag added", name=flag.name, status=flag.status.value)

    def remove_flag(self, flag_name: str) -> None:
        """
        Remove a feature flag.

        Args:
            flag_name: Name of flag to remove
        """
        if flag_name in self.flags:
            del self.flags[flag_name]
            logger.info("Feature flag removed", flag=flag_name)

    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """
        Get feature flag configuration.

        Args:
            flag_name: Name of flag

        Returns:
            FeatureFlag if exists, None otherwise
        """
        return self.flags.get(flag_name)

    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """
        Get all feature flags.

        Returns:
            Dictionary of all flags
        """
        return self.flags.copy()

    def get_enabled_flags(self) -> List[str]:
        """
        Get list of fully enabled flags.

        Returns:
            List of enabled flag names
        """
        return [
            name for name, flag in self.flags.items()
            if flag.status == FlagStatus.ENABLED
        ]

    def get_analytics(self) -> Dict[str, Any]:
        """
        Get feature flag analytics.

        Returns:
            Dictionary with evaluation statistics
        """
        analytics = {}

        for flag_name in self.flags.keys():
            evaluations = self._evaluation_counts.get(flag_name, 0)
            enabled = self._enabled_counts.get(flag_name, 0)

            analytics[flag_name] = {
                'evaluations': evaluations,
                'enabled_count': enabled,
                'enabled_rate': enabled / evaluations if evaluations > 0 else 0
            }

        return analytics

    async def refresh_flags(self, fetch_function: Callable[[], Dict[str, Any]]) -> None:
        """
        Refresh flags from remote source.

        Args:
            fetch_function: Async function that fetches flag configuration

        Example:
            >>> async def fetch_from_api():
            ...     return await api_client.get_feature_flags()
            >>> await flag_mgr.refresh_flags(fetch_from_api)
        """
        try:
            flags_config = await fetch_function()
            self._load_flags(flags_config)
            logger.info("Feature flags refreshed", count=len(flags_config))

        except Exception as e:
            logger.error("Failed to refresh feature flags", error=str(e))

    def to_dict(self) -> Dict[str, Any]:
        """
        Export flags as dictionary.

        Returns:
            Dictionary representation of all flags
        """
        return {
            name: {
                'status': flag.status.value,
                'description': flag.description,
                'percentage': flag.percentage,
                'conditions': flag.conditions,
                'enabled_for': list(flag.enabled_for),
                'disabled_for': list(flag.disabled_for),
                'metadata': flag.metadata
            }
            for name, flag in self.flags.items()
        }
