"""Feature store for managing and serving ML features.

This module provides a production-ready feature store implementation for
storing, versioning, and serving features for machine learning models.
Supports both batch and streaming feature computation with Polars DataFrames.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone
from pathlib import Path
import polars as pl
from structlog import get_logger
import aiofiles
import pickle
from collections import OrderedDict

logger = get_logger(__name__)


class FeatureStoreError(Exception):
    """Base exception for feature store operations."""
    pass


class FeatureNotFoundError(FeatureStoreError):
    """Raised when requested feature does not exist."""
    pass


class FeatureVersionError(FeatureStoreError):
    """Raised when feature version mismatch occurs."""
    pass


class FeatureStore:
    """Production-ready feature store for ML features.

    Manages feature storage, versioning, retrieval, and serving with
    support for both batch and real-time feature computation.

    Attributes:
        config: Configuration dictionary containing store parameters
        cache: In-memory LRU cache for frequently accessed features
        metadata: Feature metadata and versioning information
        storage_path: Path to persistent feature storage

    Example:
        >>> config = {
        ...     'storage_path': '/data/features',
        ...     'cache_size': 1000,
        ...     'enable_versioning': True,
        ...     'compression': 'zstd'
        ... }
        >>> store = FeatureStore(config)
        >>> await store.initialize()
        >>> features = await store.get_features('BTC/USDT', ['rsi', 'macd'])
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize feature store.

        Args:
            config: Configuration dictionary with keys:
                - storage_path: Path for persistent storage
                - cache_size: Maximum cache entries
                - enable_versioning: Enable feature versioning
                - compression: Compression algorithm
                - ttl_seconds: Feature TTL in seconds

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.storage_path = Path(self.config['storage_path'])
        self.cache_size = self.config.get('cache_size', 1000)
        self.enable_versioning = self.config.get('enable_versioning', True)
        self.compression = self.config.get('compression', 'zstd')
        self.ttl_seconds = self.config.get('ttl_seconds', 3600)

        self.cache: OrderedDict = OrderedDict()
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

        logger.info(
            "feature_store_initialized",
            storage_path=str(self.storage_path),
            cache_size=self.cache_size,
            compression=self.compression
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config keys are missing or invalid
        """
        required_keys = ['storage_path']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if 'cache_size' in self.config and self.config['cache_size'] < 1:
            raise ValueError("cache_size must be positive")

        if 'ttl_seconds' in self.config and self.config['ttl_seconds'] < 0:
            raise ValueError("ttl_seconds cannot be negative")

    async def initialize(self) -> None:
        """Initialize storage and load metadata.

        Raises:
            FeatureStoreError: If initialization fails
        """
        try:
            async with self._lock:
                if self._initialized:
                    logger.warning("feature_store_already_initialized")
                    return

                # Create storage directory
                self.storage_path.mkdir(parents=True, exist_ok=True)

                # Load metadata if exists
                metadata_path = self.storage_path / "metadata.pkl"
                if metadata_path.exists():
                    async with aiofiles.open(metadata_path, 'rb') as f:
                        content = await f.read()
                        self.metadata = pickle.loads(content)

                    logger.info(
                        "feature_metadata_loaded",
                        feature_count=len(self.metadata)
                    )

                self._initialized = True

                logger.info("feature_store_ready")

        except Exception as e:
            logger.error("feature_store_init_failed", error=str(e))
            raise FeatureStoreError(f"Initialization failed: {str(e)}") from e

    async def store_features(
        self,
        feature_group: str,
        features: pl.DataFrame,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store features in the feature store.

        Args:
            feature_group: Name of the feature group
            features: Polars DataFrame containing features
            version: Optional version string (auto-generated if None)
            metadata: Optional metadata dictionary

        Returns:
            Version string of stored features

        Raises:
            FeatureStoreError: If storage fails
            ValueError: If features DataFrame is invalid

        Example:
            >>> df = pl.DataFrame({
            ...     'timestamp': [datetime.now(timezone.utc)],
            ...     'symbol': ['BTC/USDT'],
            ...     'rsi': [Decimal('45.5')]
            ... })
            >>> version = await store.store_features('technical', df)
        """
        if not self._initialized:
            raise FeatureStoreError("Feature store not initialized")

        if features.is_empty():
            raise ValueError("Cannot store empty DataFrame")

        if 'timestamp' not in features.columns:
            raise ValueError("Features must contain 'timestamp' column")

        try:
            async with self._lock:
                # Generate version if not provided
                if version is None:
                    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

                # Create feature group directory
                group_path = self.storage_path / feature_group
                group_path.mkdir(parents=True, exist_ok=True)

                # Store features using Polars native format
                feature_path = group_path / f"{version}.parquet"
                features.write_parquet(
                    feature_path,
                    compression=self.compression,
                    use_pyarrow=False
                )

                # Update metadata
                feature_key = f"{feature_group}:{version}"
                self.metadata[feature_key] = {
                    'feature_group': feature_group,
                    'version': version,
                    'timestamp': datetime.now(timezone.utc),
                    'row_count': features.height,
                    'columns': features.columns,
                    'schema': {col: str(dtype) for col, dtype in features.schema.items()},
                    'metadata': metadata or {}
                }

                # Save metadata
                await self._save_metadata()

                # Update cache
                self._update_cache(feature_key, features)

                logger.info(
                    "features_stored",
                    feature_group=feature_group,
                    version=version,
                    row_count=features.height,
                    columns=len(features.columns)
                )

                return version

        except Exception as e:
            logger.error(
                "feature_store_failed",
                feature_group=feature_group,
                error=str(e)
            )
            raise FeatureStoreError(f"Failed to store features: {str(e)}") from e

    async def get_features(
        self,
        feature_group: str,
        feature_names: Optional[List[str]] = None,
        version: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pl.DataFrame:
        """Retrieve features from the feature store.

        Args:
            feature_group: Name of the feature group
            feature_names: Optional list of specific features to retrieve
            version: Optional specific version (uses latest if None)
            start_time: Optional start time filter (UTC)
            end_time: Optional end time filter (UTC)

        Returns:
            Polars DataFrame containing requested features

        Raises:
            FeatureNotFoundError: If feature group not found
            FeatureStoreError: If retrieval fails

        Example:
            >>> features = await store.get_features(
            ...     'technical',
            ...     feature_names=['rsi', 'macd'],
            ...     start_time=datetime(2024, 1, 1, tzinfo=timezone.utc)
            ... )
        """
        if not self._initialized:
            raise FeatureStoreError("Feature store not initialized")

        try:
            # Determine version to use
            if version is None:
                version = await self._get_latest_version(feature_group)

            if version is None:
                raise FeatureNotFoundError(
                    f"No features found for group: {feature_group}"
                )

            feature_key = f"{feature_group}:{version}"

            # Check cache first
            if feature_key in self.cache:
                logger.debug("feature_cache_hit", feature_key=feature_key)
                features = self.cache[feature_key]
            else:
                # Load from disk
                feature_path = self.storage_path / feature_group / f"{version}.parquet"
                if not feature_path.exists():
                    raise FeatureNotFoundError(
                        f"Feature file not found: {feature_path}"
                    )

                features = pl.read_parquet(feature_path, use_pyarrow=False)
                self._update_cache(feature_key, features)

                logger.debug("feature_loaded_from_disk", feature_key=feature_key)

            # Apply filters
            if start_time is not None:
                features = features.filter(pl.col('timestamp') >= start_time)

            if end_time is not None:
                features = features.filter(pl.col('timestamp') <= end_time)

            # Select specific features if requested
            if feature_names is not None:
                # Include timestamp and any identifier columns
                base_cols = ['timestamp']
                if 'symbol' in features.columns:
                    base_cols.append('symbol')

                select_cols = base_cols + [
                    col for col in feature_names if col in features.columns
                ]
                features = features.select(select_cols)

            logger.info(
                "features_retrieved",
                feature_group=feature_group,
                version=version,
                row_count=features.height,
                columns=len(features.columns)
            )

            return features

        except FeatureNotFoundError:
            raise
        except Exception as e:
            logger.error(
                "feature_retrieval_failed",
                feature_group=feature_group,
                error=str(e)
            )
            raise FeatureStoreError(f"Failed to retrieve features: {str(e)}") from e

    async def list_feature_groups(self) -> List[str]:
        """List all available feature groups.

        Returns:
            List of feature group names
        """
        if not self._initialized:
            raise FeatureStoreError("Feature store not initialized")

        try:
            groups = set()
            for feature_key in self.metadata.keys():
                group = feature_key.split(':')[0]
                groups.add(group)

            return sorted(list(groups))

        except Exception as e:
            logger.error("list_groups_failed", error=str(e))
            raise FeatureStoreError(f"Failed to list groups: {str(e)}") from e

    async def list_versions(self, feature_group: str) -> List[str]:
        """List all versions for a feature group.

        Args:
            feature_group: Name of the feature group

        Returns:
            List of version strings, sorted newest first
        """
        if not self._initialized:
            raise FeatureStoreError("Feature store not initialized")

        try:
            versions = []
            for feature_key, meta in self.metadata.items():
                if meta['feature_group'] == feature_group:
                    versions.append((meta['version'], meta['timestamp']))

            # Sort by timestamp, newest first
            versions.sort(key=lambda x: x[1], reverse=True)

            return [v[0] for v in versions]

        except Exception as e:
            logger.error("list_versions_failed", error=str(e))
            raise FeatureStoreError(f"Failed to list versions: {str(e)}") from e

    async def get_metadata(self, feature_group: str, version: str) -> Dict[str, Any]:
        """Get metadata for a specific feature version.

        Args:
            feature_group: Name of the feature group
            version: Version string

        Returns:
            Metadata dictionary

        Raises:
            FeatureNotFoundError: If feature version not found
        """
        if not self._initialized:
            raise FeatureStoreError("Feature store not initialized")

        feature_key = f"{feature_group}:{version}"

        if feature_key not in self.metadata:
            raise FeatureNotFoundError(
                f"Metadata not found for: {feature_key}"
            )

        return self.metadata[feature_key].copy()

    async def delete_features(
        self,
        feature_group: str,
        version: Optional[str] = None
    ) -> None:
        """Delete features from the store.

        Args:
            feature_group: Name of the feature group
            version: Optional version to delete (deletes all if None)

        Raises:
            FeatureStoreError: If deletion fails
        """
        if not self._initialized:
            raise FeatureStoreError("Feature store not initialized")

        try:
            async with self._lock:
                if version is None:
                    # Delete entire feature group
                    group_path = self.storage_path / feature_group
                    if group_path.exists():
                        import shutil
                        shutil.rmtree(group_path)

                    # Remove from metadata and cache
                    keys_to_remove = [
                        k for k in self.metadata.keys()
                        if k.startswith(f"{feature_group}:")
                    ]
                    for key in keys_to_remove:
                        del self.metadata[key]
                        self.cache.pop(key, None)

                    logger.info("feature_group_deleted", feature_group=feature_group)
                else:
                    # Delete specific version
                    feature_path = self.storage_path / feature_group / f"{version}.parquet"
                    if feature_path.exists():
                        feature_path.unlink()

                    feature_key = f"{feature_group}:{version}"
                    self.metadata.pop(feature_key, None)
                    self.cache.pop(feature_key, None)

                    logger.info(
                        "feature_version_deleted",
                        feature_group=feature_group,
                        version=version
                    )

                await self._save_metadata()

        except Exception as e:
            logger.error("feature_deletion_failed", error=str(e))
            raise FeatureStoreError(f"Failed to delete features: {str(e)}") from e

    async def _get_latest_version(self, feature_group: str) -> Optional[str]:
        """Get latest version for a feature group.

        Args:
            feature_group: Name of the feature group

        Returns:
            Latest version string or None if no versions exist
        """
        versions = await self.list_versions(feature_group)
        return versions[0] if versions else None

    def _update_cache(self, key: str, data: pl.DataFrame) -> None:
        """Update LRU cache with feature data.

        Args:
            key: Cache key
            data: Feature DataFrame
        """
        # Remove oldest if cache is full
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)

        # Add to cache (moves to end if exists)
        self.cache[key] = data
        self.cache.move_to_end(key)

    async def _save_metadata(self) -> None:
        """Save metadata to disk.

        Raises:
            FeatureStoreError: If save fails
        """
        try:
            metadata_path = self.storage_path / "metadata.pkl"
            async with aiofiles.open(metadata_path, 'wb') as f:
                await f.write(pickle.dumps(self.metadata))

            logger.debug("metadata_saved", feature_count=len(self.metadata))

        except Exception as e:
            logger.error("metadata_save_failed", error=str(e))
            raise FeatureStoreError(f"Failed to save metadata: {str(e)}") from e

    async def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        async with self._lock:
            self.cache.clear()
            logger.info("feature_cache_cleared")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get feature store statistics.

        Returns:
            Dictionary containing store statistics
        """
        if not self._initialized:
            raise FeatureStoreError("Feature store not initialized")

        try:
            groups = await self.list_feature_groups()

            total_versions = len(self.metadata)
            total_features = sum(
                len(meta['columns']) for meta in self.metadata.values()
            )

            cache_hit_rate = Decimal('0.0')  # Would track in production

            stats = {
                'total_feature_groups': len(groups),
                'total_versions': total_versions,
                'total_features': total_features,
                'cache_size': len(self.cache),
                'cache_capacity': self.cache_size,
                'cache_hit_rate': cache_hit_rate,
                'storage_path': str(self.storage_path),
                'compression': self.compression
            }

            return stats

        except Exception as e:
            logger.error("get_statistics_failed", error=str(e))
            raise FeatureStoreError(f"Failed to get statistics: {str(e)}") from e
