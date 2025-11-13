"""
Model Registry for Version Control and Model Management.

This module provides a centralized registry for managing ML models,
their versions, metadata, and lifecycle operations.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import json
import shutil

import polars as pl
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel
from quantum_trader.exceptions import RegistryError, ValidationError

logger = get_logger(__name__)


class ModelStatus(Enum):
    """Model status types."""
    DEVELOPMENT = "DEVELOPMENT"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"
    DEPRECATED = "DEPRECATED"


@dataclass
class ModelMetadata:
    """Metadata for registered model.

    Attributes:
        model_id: Unique model identifier
        model_name: Model name
        version: Model version
        status: Model status
        model_type: Type of model (e.g., 'lightgbm', 'informer')
        created_at: Creation timestamp
        updated_at: Last update timestamp
        created_by: Creator identifier
        description: Model description
        tags: Model tags
        metrics: Performance metrics
        parameters: Model parameters
        dependencies: Model dependencies
        path: Storage path
        metadata: Additional metadata
    """
    model_id: str
    model_name: str
    version: str
    status: ModelStatus
    model_type: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, Decimal] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistryConfig:
    """Configuration for model registry.

    Attributes:
        registry_path: Path to registry storage
        versioning_scheme: Version numbering scheme
        max_versions: Maximum versions to keep per model
        auto_archive: Whether to auto-archive old versions
        enable_metrics_tracking: Whether to track metrics history
    """
    registry_path: str
    versioning_scheme: str = "semantic"  # semantic, timestamp, sequential
    max_versions: int = 10
    auto_archive: bool = True
    enable_metrics_tracking: bool = True


class ModelRegistry:
    """Centralized registry for ML model management.

    Manages model versions, metadata, lifecycle, and deployment status
    with full version control and audit trail.

    Attributes:
        config: Registry configuration
        registry_path: Path to registry storage
        models: Dictionary of registered models

    Example:
        >>> config = RegistryConfig(
        ...     registry_path="/models/registry",
        ...     versioning_scheme="semantic",
        ...     max_versions=10
        ... )
        >>> registry = ModelRegistry(config)
        >>> model_id = registry.register_model(
        ...     model,
        ...     name="price_predictor",
        ...     version="1.0.0",
        ...     model_type="lightgbm"
        ... )
        >>> registry.promote_to_production(model_id)
    """

    def __init__(self, config: RegistryConfig) -> None:
        """Initialize model registry.

        Args:
            config: Registry configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.registry_path = Path(config.registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, ModelMetadata] = {}
        self._load_registry()

        logger.info(
            "Model registry initialized",
            registry_path=str(self.registry_path),
            num_models=len(self.models)
        )

    def _validate_config(self) -> None:
        """Validate registry configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        if not self.config.registry_path:
            raise ValidationError("registry_path cannot be empty")

        valid_schemes = ["semantic", "timestamp", "sequential"]
        if self.config.versioning_scheme not in valid_schemes:
            raise ValidationError(f"Invalid versioning_scheme: {self.config.versioning_scheme}")

        if self.config.max_versions < 1:
            raise ValidationError("max_versions must be >= 1")

    def _load_registry(self) -> None:
        """Load registry from disk."""
        try:
            registry_file = self.registry_path / "registry.json"

            if registry_file.exists():
                with open(registry_file, "r") as f:
                    data = json.load(f)

                for model_id, model_data in data.items():
                    # Convert datetime strings back to datetime objects
                    model_data["created_at"] = datetime.fromisoformat(model_data["created_at"])
                    model_data["updated_at"] = datetime.fromisoformat(model_data["updated_at"])
                    model_data["status"] = ModelStatus(model_data["status"])

                    # Convert metrics to Decimal
                    if "metrics" in model_data:
                        model_data["metrics"] = {
                            k: Decimal(str(v)) for k, v in model_data["metrics"].items()
                        }

                    self.models[model_id] = ModelMetadata(**model_data)

                logger.info("Registry loaded", num_models=len(self.models))

        except Exception as e:
            logger.warning("Failed to load registry", error=str(e))

    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            registry_file = self.registry_path / "registry.json"

            data = {}
            for model_id, metadata in self.models.items():
                model_dict = asdict(metadata)

                # Convert datetime to ISO format
                model_dict["created_at"] = metadata.created_at.isoformat()
                model_dict["updated_at"] = metadata.updated_at.isoformat()
                model_dict["status"] = metadata.status.value

                # Convert Decimal to float for JSON
                if "metrics" in model_dict:
                    model_dict["metrics"] = {
                        k: float(v) for k, v in metadata.metrics.items()
                    }

                data[model_id] = model_dict

            with open(registry_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Registry saved")

        except Exception as e:
            logger.error("Failed to save registry", error=str(e))
            raise RegistryError(f"Failed to save registry: {e}") from e

    def register_model(
        self,
        model: BaseMLModel,
        name: str,
        version: str,
        model_type: str,
        created_by: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Decimal]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new model.

        Args:
            model: Model instance to register
            name: Model name
            version: Model version
            model_type: Type of model
            created_by: Creator identifier
            description: Model description
            tags: Model tags
            metrics: Performance metrics
            parameters: Model parameters

        Returns:
            Model ID

        Raises:
            RegistryError: If registration fails
        """
        try:
            # Generate model ID
            model_id = f"{name}_{version}_{datetime.now(timezone.utc).timestamp()}"

            # Check if model already exists
            if model_id in self.models:
                raise RegistryError(f"Model already registered: {model_id}")

            # Create model directory
            model_path = self.registry_path / model_id
            model_path.mkdir(parents=True, exist_ok=True)

            # Save model
            model_file = model_path / "model.pkl"
            model.save(str(model_file))

            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=name,
                version=version,
                status=ModelStatus.DEVELOPMENT,
                model_type=model_type,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                created_by=created_by,
                description=description,
                tags=tags or [],
                metrics=metrics or {},
                parameters=parameters or {},
                path=str(model_path)
            )

            # Register model
            self.models[model_id] = metadata
            self._save_registry()

            # Clean up old versions if needed
            if self.config.auto_archive:
                self._cleanup_old_versions(name)

            logger.info(
                "Model registered",
                model_id=model_id,
                name=name,
                version=version
            )

            return model_id

        except Exception as e:
            logger.error("Model registration failed", error=str(e))
            raise RegistryError(f"Model registration failed: {e}") from e

    def get_model(
        self,
        model_id: Optional[str] = None,
        name: Optional[str] = None,
        version: Optional[str] = None,
        status: Optional[ModelStatus] = None
    ) -> Optional[ModelMetadata]:
        """Get model metadata.

        Args:
            model_id: Model ID
            name: Model name
            version: Model version
            status: Model status

        Returns:
            Model metadata or None

        Raises:
            ValidationError: If search criteria are invalid
        """
        try:
            # Search by ID
            if model_id is not None:
                return self.models.get(model_id)

            # Search by name, version, status
            candidates = []
            for metadata in self.models.values():
                if name is not None and metadata.model_name != name:
                    continue
                if version is not None and metadata.version != version:
                    continue
                if status is not None and metadata.status != status:
                    continue
                candidates.append(metadata)

            # Return most recent
            if candidates:
                return max(candidates, key=lambda m: m.updated_at)

            return None

        except Exception as e:
            logger.error("Model retrieval failed", error=str(e))
            raise RegistryError(f"Model retrieval failed: {e}") from e

    def list_models(
        self,
        name: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """List registered models.

        Args:
            name: Filter by model name
            status: Filter by status
            tags: Filter by tags

        Returns:
            List of model metadata
        """
        try:
            results = []

            for metadata in self.models.values():
                # Apply filters
                if name is not None and metadata.model_name != name:
                    continue
                if status is not None and metadata.status != status:
                    continue
                if tags is not None and not all(tag in metadata.tags for tag in tags):
                    continue

                results.append(metadata)

            # Sort by updated_at descending
            results.sort(key=lambda m: m.updated_at, reverse=True)

            logger.info("Models listed", num_results=len(results))

            return results

        except Exception as e:
            logger.error("Model listing failed", error=str(e))
            raise RegistryError(f"Model listing failed: {e}") from e

    def update_status(
        self,
        model_id: str,
        new_status: ModelStatus
    ) -> None:
        """Update model status.

        Args:
            model_id: Model ID
            new_status: New status

        Raises:
            RegistryError: If update fails
        """
        try:
            if model_id not in self.models:
                raise RegistryError(f"Model not found: {model_id}")

            metadata = self.models[model_id]
            old_status = metadata.status

            metadata.status = new_status
            metadata.updated_at = datetime.now(timezone.utc)

            self._save_registry()

            logger.info(
                "Model status updated",
                model_id=model_id,
                old_status=old_status.value,
                new_status=new_status.value
            )

        except Exception as e:
            logger.error("Status update failed", error=str(e))
            raise RegistryError(f"Status update failed: {e}") from e

    def promote_to_production(self, model_id: str) -> None:
        """Promote model to production.

        Args:
            model_id: Model ID

        Raises:
            RegistryError: If promotion fails
        """
        try:
            if model_id not in self.models:
                raise RegistryError(f"Model not found: {model_id}")

            metadata = self.models[model_id]

            # Demote current production models of same name
            for other_id, other_metadata in self.models.items():
                if (other_metadata.model_name == metadata.model_name and
                        other_metadata.status == ModelStatus.PRODUCTION):
                    self.update_status(other_id, ModelStatus.STAGING)

            # Promote to production
            self.update_status(model_id, ModelStatus.PRODUCTION)

            logger.info("Model promoted to production", model_id=model_id)

        except Exception as e:
            logger.error("Promotion failed", error=str(e))
            raise RegistryError(f"Promotion failed: {e}") from e

    def archive_model(self, model_id: str) -> None:
        """Archive a model.

        Args:
            model_id: Model ID

        Raises:
            RegistryError: If archival fails
        """
        try:
            if model_id not in self.models:
                raise RegistryError(f"Model not found: {model_id}")

            self.update_status(model_id, ModelStatus.ARCHIVED)

            logger.info("Model archived", model_id=model_id)

        except Exception as e:
            logger.error("Archival failed", error=str(e))
            raise RegistryError(f"Archival failed: {e}") from e

    def delete_model(self, model_id: str) -> None:
        """Delete a model.

        Args:
            model_id: Model ID

        Raises:
            RegistryError: If deletion fails
        """
        try:
            if model_id not in self.models:
                raise RegistryError(f"Model not found: {model_id}")

            metadata = self.models[model_id]

            # Prevent deletion of production models
            if metadata.status == ModelStatus.PRODUCTION:
                raise RegistryError("Cannot delete production model")

            # Delete model files
            if metadata.path:
                model_path = Path(metadata.path)
                if model_path.exists():
                    shutil.rmtree(model_path)

            # Remove from registry
            del self.models[model_id]
            self._save_registry()

            logger.info("Model deleted", model_id=model_id)

        except Exception as e:
            logger.error("Deletion failed", error=str(e))
            raise RegistryError(f"Deletion failed: {e}") from e

    def update_metrics(
        self,
        model_id: str,
        metrics: Dict[str, Decimal]
    ) -> None:
        """Update model metrics.

        Args:
            model_id: Model ID
            metrics: New metrics

        Raises:
            RegistryError: If update fails
        """
        try:
            if model_id not in self.models:
                raise RegistryError(f"Model not found: {model_id}")

            metadata = self.models[model_id]
            metadata.metrics.update(metrics)
            metadata.updated_at = datetime.now(timezone.utc)

            self._save_registry()

            logger.info("Model metrics updated", model_id=model_id)

        except Exception as e:
            logger.error("Metrics update failed", error=str(e))
            raise RegistryError(f"Metrics update failed: {e}") from e

    def _cleanup_old_versions(self, model_name: str) -> None:
        """Clean up old versions of a model.

        Args:
            model_name: Model name
        """
        try:
            # Get all versions of the model
            versions = [
                (model_id, metadata)
                for model_id, metadata in self.models.items()
                if metadata.model_name == model_name
            ]

            # Sort by created_at
            versions.sort(key=lambda x: x[1].created_at, reverse=True)

            # Archive old versions beyond max_versions
            for model_id, metadata in versions[self.config.max_versions:]:
                if metadata.status != ModelStatus.PRODUCTION:
                    self.archive_model(model_id)

            logger.debug(
                "Old versions cleaned up",
                model_name=model_name,
                num_kept=min(len(versions), self.config.max_versions)
            )

        except Exception as e:
            logger.warning("Cleanup failed", error=str(e))

    async def register_model_async(
        self,
        model: BaseMLModel,
        name: str,
        version: str,
        model_type: str,
        created_by: str,
        **kwargs
    ) -> str:
        """Register model asynchronously."""
        return await asyncio.to_thread(
            self.register_model,
            model,
            name,
            version,
            model_type,
            created_by,
            **kwargs
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_models": len(self.models),
            "by_status": {},
            "by_type": {}
        }

        # Count by status
        for status in ModelStatus:
            count = sum(1 for m in self.models.values() if m.status == status)
            stats["by_status"][status.value] = count

        # Count by type
        types: Dict[str, int] = {}
        for metadata in self.models.values():
            types[metadata.model_type] = types.get(metadata.model_type, 0) + 1
        stats["by_type"] = types

        return stats
