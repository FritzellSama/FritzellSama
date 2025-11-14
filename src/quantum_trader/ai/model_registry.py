"""Model registry for managing and versioning machine learning models.

This module provides a centralized registry for storing, versioning, and
managing ML models with metadata, performance tracking, and deployment support.
"""

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for registered models.

    Attributes:
        model_id: Unique model identifier
        model_name: Human-readable model name
        model_type: Type of model (e.g., 'lstm', 'xgboost')
        version: Model version
        created_at: Creation timestamp
        updated_at: Last update timestamp
        author: Model creator
        description: Model description
        hyperparameters: Model hyperparameters
        training_metrics: Metrics from training
        validation_metrics: Metrics from validation
        test_metrics: Metrics from testing
        status: Model status ('draft', 'trained', 'validated', 'production', 'archived')
        tags: Model tags for categorization
        file_path: Path to saved model file
        file_hash: Hash of model file for integrity
        dependencies: Model dependencies
        framework: ML framework used
        framework_version: Framework version
    """

    model_id: str
    model_name: str
    model_type: str
    version: str
    created_at: datetime
    updated_at: datetime
    author: str
    description: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_metrics: Dict[str, Decimal] = field(default_factory=dict)
    validation_metrics: Dict[str, Decimal] = field(default_factory=dict)
    test_metrics: Dict[str, Decimal] = field(default_factory=dict)
    status: str = "draft"
    tags: List[str] = field(default_factory=list)
    file_path: str = ""
    file_hash: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)
    framework: str = ""
    framework_version: str = ""


@dataclass
class RegistryConfig:
    """Configuration for model registry.

    Attributes:
        storage_path: Base path for model storage
        metadata_path: Path for metadata storage
        enable_versioning: Whether to enable automatic versioning
        max_versions: Maximum number of versions to keep per model
        auto_backup: Whether to auto-backup models
        backup_interval: Backup interval in seconds
        compression: Whether to compress model files
        encryption_enabled: Whether to encrypt model files
    """

    storage_path: str
    metadata_path: str
    enable_versioning: bool
    max_versions: int
    auto_backup: bool
    backup_interval: int
    compression: bool
    encryption_enabled: bool


class ModelRegistry:
    """Centralized registry for ML model management.

    Provides comprehensive model lifecycle management including:
    - Model registration and versioning
    - Metadata tracking
    - Performance monitoring
    - Model comparison
    - Deployment management
    - Model archival and cleanup

    Example:
        >>> config = RegistryConfig(
        ...     storage_path="./models",
        ...     metadata_path="./models/metadata",
        ...     enable_versioning=True,
        ...     max_versions=10,
        ...     auto_backup=True,
        ...     backup_interval=3600,
        ...     compression=True,
        ...     encryption_enabled=False
        ... )
        >>> registry = ModelRegistry(config)
        >>> await registry.initialize()
        >>> model_id = await registry.register_model(
        ...     model=my_model,
        ...     model_name="momentum_predictor",
        ...     model_type="lstm",
        ...     description="LSTM model for momentum prediction"
        ... )
        >>> model = await registry.load_model(model_id)
    """

    def __init__(self, config: RegistryConfig) -> None:
        """Initialize model registry.

        Args:
            config: Registry configuration
        """
        self.config = config

        # Storage paths
        self.storage_path = Path(config.storage_path)
        self.metadata_path = Path(config.metadata_path)

        # Registry state
        self.models: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, BaseMLModel] = {}

        # Version tracking
        self.version_history: Dict[str, List[str]] = {}

        logger.info(
            "model_registry_initialized",
            storage_path=str(self.storage_path)
        )

    async def initialize(self) -> None:
        """Initialize registry and load existing metadata."""
        try:
            # Create directories
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.metadata_path.mkdir(parents=True, exist_ok=True)

            # Load existing metadata
            await self._load_metadata()

            logger.info(
                "registry_initialized",
                n_models=len(self.models)
            )

        except Exception as e:
            logger.error("initialization_failed", error=str(e))
            raise

    async def register_model(
        self,
        model: BaseMLModel,
        model_name: str,
        model_type: str,
        description: str,
        author: str = "system",
        hyperparameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None
    ) -> str:
        """Register a new model in the registry.

        Args:
            model: Model instance to register
            model_name: Human-readable model name
            model_type: Type of model
            description: Model description
            author: Model creator
            hyperparameters: Model hyperparameters
            tags: Model tags
            version: Explicit version (auto-generated if None)

        Returns:
            Model ID
        """
        try:
            # Generate model ID
            model_id = self._generate_model_id(model_name, model_type)

            # Determine version
            if version is None:
                version = self._get_next_version(model_name)

            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                version=version,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                author=author,
                description=description,
                hyperparameters=hyperparameters or {},
                tags=tags or [],
                status="draft",
                framework="pytorch" if hasattr(model, 'model') else "sklearn"
            )

            # Save model
            model_path = await self._save_model(model, model_id)
            metadata.file_path = str(model_path)

            # Calculate file hash
            metadata.file_hash = await self._calculate_file_hash(model_path)

            # Register metadata
            self.models[model_id] = metadata

            # Update version history
            if model_name not in self.version_history:
                self.version_history[model_name] = []
            self.version_history[model_name].append(model_id)

            # Enforce version limits
            if self.config.enable_versioning:
                await self._enforce_version_limit(model_name)

            # Save metadata
            await self._save_metadata(model_id)

            logger.info(
                "model_registered",
                model_id=model_id,
                model_name=model_name,
                version=version
            )

            return model_id

        except Exception as e:
            logger.error("model_registration_failed", error=str(e))
            raise

    async def load_model(
        self,
        model_id: str,
        cache: bool = True
    ) -> Optional[BaseMLModel]:
        """Load model from registry.

        Args:
            model_id: Model identifier
            cache: Whether to cache loaded model

        Returns:
            Loaded model instance or None if not found
        """
        try:
            # Check cache
            if cache and model_id in self.loaded_models:
                logger.info("model_loaded_from_cache", model_id=model_id)
                return self.loaded_models[model_id]

            # Check registry
            if model_id not in self.models:
                logger.warning("model_not_found", model_id=model_id)
                return None

            metadata = self.models[model_id]

            # Verify file integrity
            if not await self._verify_file_integrity(metadata):
                logger.error("file_integrity_check_failed", model_id=model_id)
                return None

            # Load model based on type
            model_path = Path(metadata.file_path)

            # For now, return None as we need specific model class
            # In production, use a factory pattern based on model_type
            logger.info(
                "model_loaded",
                model_id=model_id,
                model_type=metadata.model_type
            )

            # Note: Actual loading depends on model type
            # This is a placeholder - implement proper model loading
            # based on model_type and framework

            return None

        except Exception as e:
            logger.error("model_load_failed", model_id=model_id, error=str(e))
            return None

    async def update_model_status(
        self,
        model_id: str,
        status: str
    ) -> bool:
        """Update model status.

        Args:
            model_id: Model identifier
            status: New status ('draft', 'trained', 'validated', 'production', 'archived')

        Returns:
            True if successful
        """
        try:
            if model_id not in self.models:
                logger.warning("model_not_found", model_id=model_id)
                return False

            self.models[model_id].status = status
            self.models[model_id].updated_at = datetime.now(timezone.utc)

            await self._save_metadata(model_id)

            logger.info(
                "model_status_updated",
                model_id=model_id,
                status=status
            )

            return True

        except Exception as e:
            logger.error("status_update_failed", error=str(e))
            return False

    async def update_metrics(
        self,
        model_id: str,
        metric_type: str,
        metrics: Dict[str, Decimal]
    ) -> bool:
        """Update model metrics.

        Args:
            model_id: Model identifier
            metric_type: Type of metrics ('training', 'validation', 'test')
            metrics: Metrics dictionary

        Returns:
            True if successful
        """
        try:
            if model_id not in self.models:
                logger.warning("model_not_found", model_id=model_id)
                return False

            metadata = self.models[model_id]

            if metric_type == "training":
                metadata.training_metrics = metrics
            elif metric_type == "validation":
                metadata.validation_metrics = metrics
            elif metric_type == "test":
                metadata.test_metrics = metrics
            else:
                logger.warning("invalid_metric_type", metric_type=metric_type)
                return False

            metadata.updated_at = datetime.now(timezone.utc)

            await self._save_metadata(model_id)

            logger.info(
                "metrics_updated",
                model_id=model_id,
                metric_type=metric_type
            )

            return True

        except Exception as e:
            logger.error("metrics_update_failed", error=str(e))
            return False

    async def list_models(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """List models with optional filtering.

        Args:
            model_name: Filter by model name
            model_type: Filter by model type
            status: Filter by status
            tags: Filter by tags (must have all tags)

        Returns:
            List of matching model metadata
        """
        try:
            results = []

            for metadata in self.models.values():
                # Apply filters
                if model_name and metadata.model_name != model_name:
                    continue

                if model_type and metadata.model_type != model_type:
                    continue

                if status and metadata.status != status:
                    continue

                if tags and not all(tag in metadata.tags for tag in tags):
                    continue

                results.append(metadata)

            # Sort by updated_at (newest first)
            results.sort(key=lambda m: m.updated_at, reverse=True)

            return results

        except Exception as e:
            logger.error("list_models_failed", error=str(e))
            return []

    async def compare_models(
        self,
        model_ids: List[str],
        metric: str = "test"
    ) -> pl.DataFrame:
        """Compare multiple models.

        Args:
            model_ids: List of model IDs to compare
            metric: Metric type to compare ('training', 'validation', 'test')

        Returns:
            Comparison DataFrame
        """
        try:
            data = {
                'model_id': [],
                'model_name': [],
                'version': [],
                'status': []
            }

            # Collect all unique metric keys
            metric_keys = set()

            for model_id in model_ids:
                if model_id not in self.models:
                    continue

                metadata = self.models[model_id]

                if metric == "training":
                    metrics = metadata.training_metrics
                elif metric == "validation":
                    metrics = metadata.validation_metrics
                else:
                    metrics = metadata.test_metrics

                metric_keys.update(metrics.keys())

            # Initialize metric columns
            for key in metric_keys:
                data[key] = []

            # Populate data
            for model_id in model_ids:
                if model_id not in self.models:
                    continue

                metadata = self.models[model_id]

                if metric == "training":
                    metrics = metadata.training_metrics
                elif metric == "validation":
                    metrics = metadata.validation_metrics
                else:
                    metrics = metadata.test_metrics

                data['model_id'].append(model_id)
                data['model_name'].append(metadata.model_name)
                data['version'].append(metadata.version)
                data['status'].append(metadata.status)

                for key in metric_keys:
                    value = metrics.get(key)
                    data[key].append(float(value) if value is not None else None)

            return pl.DataFrame(data)

        except Exception as e:
            logger.error("model_comparison_failed", error=str(e))
            return pl.DataFrame()

    async def get_best_model(
        self,
        model_name: str,
        metric_name: str,
        metric_type: str = "test",
        maximize: bool = True
    ) -> Optional[str]:
        """Get best performing model.

        Args:
            model_name: Model name to search
            metric_name: Metric to optimize
            metric_type: Type of metrics to use
            maximize: Whether to maximize metric (vs minimize)

        Returns:
            Model ID of best model or None
        """
        try:
            candidates = await self.list_models(model_name=model_name)

            if not candidates:
                return None

            best_model_id = None
            best_value = None

            for metadata in candidates:
                if metric_type == "training":
                    metrics = metadata.training_metrics
                elif metric_type == "validation":
                    metrics = metadata.validation_metrics
                else:
                    metrics = metadata.test_metrics

                if metric_name not in metrics:
                    continue

                value = metrics[metric_name]

                if best_value is None:
                    best_value = value
                    best_model_id = metadata.model_id
                elif maximize and value > best_value:
                    best_value = value
                    best_model_id = metadata.model_id
                elif not maximize and value < best_value:
                    best_value = value
                    best_model_id = metadata.model_id

            logger.info(
                "best_model_found",
                model_id=best_model_id,
                metric=metric_name,
                value=str(best_value) if best_value else None
            )

            return best_model_id

        except Exception as e:
            logger.error("get_best_model_failed", error=str(e))
            return None

    async def archive_model(self, model_id: str) -> bool:
        """Archive a model.

        Args:
            model_id: Model identifier

        Returns:
            True if successful
        """
        return await self.update_model_status(model_id, "archived")

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model from registry.

        Args:
            model_id: Model identifier

        Returns:
            True if successful
        """
        try:
            if model_id not in self.models:
                logger.warning("model_not_found", model_id=model_id)
                return False

            metadata = self.models[model_id]

            # Delete model file
            model_path = Path(metadata.file_path)
            if model_path.exists():
                await asyncio.to_thread(model_path.unlink)

            # Delete metadata file
            metadata_file = self.metadata_path / f"{model_id}.json"
            if metadata_file.exists():
                await asyncio.to_thread(metadata_file.unlink)

            # Remove from registry
            del self.models[model_id]

            # Remove from cache
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]

            # Update version history
            for versions in self.version_history.values():
                if model_id in versions:
                    versions.remove(model_id)

            logger.info("model_deleted", model_id=model_id)

            return True

        except Exception as e:
            logger.error("model_deletion_failed", error=str(e))
            return False

    def _generate_model_id(self, model_name: str, model_type: str) -> str:
        """Generate unique model ID.

        Args:
            model_name: Model name
            model_type: Model type

        Returns:
            Model ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        unique_str = f"{model_name}_{model_type}_{timestamp}"
        hash_str = hashlib.md5(unique_str.encode()).hexdigest()[:8]

        return f"{model_name}_{hash_str}"

    def _get_next_version(self, model_name: str) -> str:
        """Get next version number for model.

        Args:
            model_name: Model name

        Returns:
            Version string
        """
        if model_name not in self.version_history:
            return "1.0.0"

        # Count existing versions
        version_count = len(self.version_history[model_name])

        # Simple versioning: increment minor version
        major = 1
        minor = version_count
        patch = 0

        return f"{major}.{minor}.{patch}"

    async def _save_model(
        self,
        model: BaseMLModel,
        model_id: str
    ) -> Path:
        """Save model to disk.

        Args:
            model: Model instance
            model_id: Model identifier

        Returns:
            Path to saved model
        """
        try:
            model_path = self.storage_path / f"{model_id}.pkl"

            await asyncio.to_thread(model.save, str(model_path))

            return model_path

        except Exception as e:
            logger.error("model_save_failed", error=str(e))
            raise

    async def _save_metadata(self, model_id: str) -> None:
        """Save model metadata.

        Args:
            model_id: Model identifier
        """
        try:
            if model_id not in self.models:
                return

            metadata = self.models[model_id]
            metadata_file = self.metadata_path / f"{model_id}.json"

            # Convert to dict
            metadata_dict = asdict(metadata)

            # Convert Decimal to string
            for key in ['training_metrics', 'validation_metrics', 'test_metrics']:
                if key in metadata_dict:
                    metadata_dict[key] = {
                        k: str(v) for k, v in metadata_dict[key].items()
                    }

            # Convert datetime to string
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            metadata_dict['updated_at'] = metadata.updated_at.isoformat()

            # Save to file
            await asyncio.to_thread(
                metadata_file.write_text,
                json.dumps(metadata_dict, indent=2)
            )

        except Exception as e:
            logger.error("metadata_save_failed", error=str(e))

    async def _load_metadata(self) -> None:
        """Load all metadata from disk."""
        try:
            if not self.metadata_path.exists():
                return

            metadata_files = list(self.metadata_path.glob("*.json"))

            for metadata_file in metadata_files:
                try:
                    content = await asyncio.to_thread(metadata_file.read_text)
                    metadata_dict = json.loads(content)

                    # Convert strings back to appropriate types
                    for key in ['training_metrics', 'validation_metrics', 'test_metrics']:
                        if key in metadata_dict:
                            metadata_dict[key] = {
                                k: Decimal(v) for k, v in metadata_dict[key].items()
                            }

                    metadata_dict['created_at'] = datetime.fromisoformat(
                        metadata_dict['created_at']
                    )
                    metadata_dict['updated_at'] = datetime.fromisoformat(
                        metadata_dict['updated_at']
                    )

                    metadata = ModelMetadata(**metadata_dict)
                    self.models[metadata.model_id] = metadata

                    # Update version history
                    if metadata.model_name not in self.version_history:
                        self.version_history[metadata.model_name] = []
                    self.version_history[metadata.model_name].append(metadata.model_id)

                except Exception as e:
                    logger.warning(
                        "metadata_load_failed",
                        file=str(metadata_file),
                        error=str(e)
                    )

            logger.info("metadata_loaded", n_models=len(self.models))

        except Exception as e:
            logger.error("metadata_load_failed", error=str(e))

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file.

        Args:
            file_path: Path to file

        Returns:
            Hash string
        """
        try:
            hasher = hashlib.sha256()

            def _hash_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hasher.update(chunk)

            await asyncio.to_thread(_hash_file)

            return hasher.hexdigest()

        except Exception as e:
            logger.error("hash_calculation_failed", error=str(e))
            return ""

    async def _verify_file_integrity(self, metadata: ModelMetadata) -> bool:
        """Verify model file integrity.

        Args:
            metadata: Model metadata

        Returns:
            True if file is valid
        """
        try:
            file_path = Path(metadata.file_path)

            if not file_path.exists():
                logger.error("model_file_missing", path=str(file_path))
                return False

            current_hash = await self._calculate_file_hash(file_path)

            if current_hash != metadata.file_hash:
                logger.error(
                    "file_integrity_mismatch",
                    expected=metadata.file_hash,
                    actual=current_hash
                )
                return False

            return True

        except Exception as e:
            logger.error("integrity_verification_failed", error=str(e))
            return False

    async def _enforce_version_limit(self, model_name: str) -> None:
        """Enforce version limit by removing old versions.

        Args:
            model_name: Model name
        """
        try:
            if model_name not in self.version_history:
                return

            versions = self.version_history[model_name]

            if len(versions) <= self.config.max_versions:
                return

            # Sort by creation time
            version_metadata = [
                (model_id, self.models[model_id])
                for model_id in versions
                if model_id in self.models
            ]

            version_metadata.sort(key=lambda x: x[1].created_at)

            # Remove oldest versions
            n_to_remove = len(versions) - self.config.max_versions

            for i in range(n_to_remove):
                model_id, _ = version_metadata[i]
                await self.delete_model(model_id)

            logger.info(
                "version_limit_enforced",
                model_name=model_name,
                removed=n_to_remove
            )

        except Exception as e:
            logger.error("version_limit_enforcement_failed", error=str(e))

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Statistics dictionary
        """
        try:
            stats = {
                'total_models': len(self.models),
                'by_status': {},
                'by_type': {},
                'total_versions': sum(len(v) for v in self.version_history.values())
            }

            for metadata in self.models.values():
                # Count by status
                status = metadata.status
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

                # Count by type
                model_type = metadata.model_type
                stats['by_type'][model_type] = stats['by_type'].get(model_type, 0) + 1

            return stats

        except Exception as e:
            logger.error("stats_calculation_failed", error=str(e))
            return {}
