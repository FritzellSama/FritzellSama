"""Base models and ORM utilities for database operations.

Provides base classes for all database models with automatic timestamp
management, validation, and serialization.
"""

from decimal import Decimal
from typing import Optional, Dict, List, Any, Type, TypeVar
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from structlog import get_logger

logger = get_logger(__name__)

T = TypeVar('T', bound='BaseModel')


@dataclass
class BaseModel(ABC):
    """Base class for all database models.

    Provides common functionality for validation, serialization,
    and timestamp management. All models must use UTC timestamps
    and Decimal for monetary values.

    Attributes:
        created_at: UTC timestamp when record created
        updated_at: UTC timestamp when record last updated
    """
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate model after initialization."""
        self._validate_timestamps()
        self.validate()

    def _validate_timestamps(self) -> None:
        """Ensure all timestamps have UTC timezone."""
        for field_name in ['created_at', 'updated_at']:
            if hasattr(self, field_name):
                timestamp = getattr(self, field_name)
                if timestamp and not isinstance(timestamp.tzinfo, type(timezone.utc)):
                    if timestamp.tzinfo is None:
                        raise ValueError(f"{field_name} must have UTC timezone")

    @abstractmethod
    def validate(self) -> None:
        """Validate model data.

        Raises:
            ValueError: If validation fails
        """
        pass

    def touch(self) -> None:
        """Update the updated_at timestamp to current UTC time."""
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for serialization.

        Returns:
            Dictionary representation with proper type conversions
        """
        data = asdict(self)

        # Convert Decimal to string
        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = str(value)
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value

        return data

    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model instance from dictionary.

        Args:
            data: Dictionary with model data

        Returns:
            Model instance
        """
        pass

    @classmethod
    def get_table_name(cls) -> str:
        """Get database table name for this model.

        Returns:
            Table name (snake_case of class name by default)
        """
        # Convert CamelCase to snake_case
        name = cls.__name__
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append('_')
            result.append(char.lower())
        return ''.join(result)

    @classmethod
    @abstractmethod
    def get_schema(cls) -> Dict[str, str]:
        """Get database schema for this model.

        Returns:
            Dictionary mapping column names to PostgreSQL types
        """
        pass


from enum import Enum


@dataclass
class TimestampedModel(BaseModel):
    """Model with automatic timestamp management.

    Extends BaseModel with additional timestamp tracking for
    audit and versioning purposes.

    Attributes:
        id: Unique identifier (set by database)
        created_at: When record was created
        updated_at: When record was last updated
        deleted_at: When record was soft-deleted (None if active)
    """
    id: Optional[str] = None
    deleted_at: Optional[datetime] = None

    def validate(self) -> None:
        """Validate timestamped model."""
        if self.deleted_at:
            if not isinstance(self.deleted_at.tzinfo, type(timezone.utc)):
                if self.deleted_at.tzinfo is None:
                    raise ValueError("deleted_at must have UTC timezone")

    def soft_delete(self) -> None:
        """Mark record as deleted without removing from database."""
        self.deleted_at = datetime.now(timezone.utc)
        self.touch()

    def restore(self) -> None:
        """Restore a soft-deleted record."""
        self.deleted_at = None
        self.touch()

    @property
    def is_deleted(self) -> bool:
        """Check if record is soft-deleted.

        Returns:
            True if deleted, False otherwise
        """
        return self.deleted_at is not None

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model from dictionary.

        Args:
            data: Dictionary with model data

        Returns:
            Model instance
        """
        # Convert ISO format strings back to datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'deleted_at' in data and data['deleted_at']:
            if isinstance(data['deleted_at'], str):
                data['deleted_at'] = datetime.fromisoformat(data['deleted_at'])

        return cls(**data)

    @classmethod
    def get_schema(cls) -> Dict[str, str]:
        """Get base schema for timestamped models.

        Returns:
            Dictionary of column names to PostgreSQL types
        """
        return {
            'id': 'UUID PRIMARY KEY DEFAULT gen_random_uuid()',
            'created_at': 'TIMESTAMPTZ NOT NULL DEFAULT NOW()',
            'updated_at': 'TIMESTAMPTZ NOT NULL DEFAULT NOW()',
            'deleted_at': 'TIMESTAMPTZ'
        }


@dataclass
class AuditedModel(TimestampedModel):
    """Model with full audit trail support.

    Extends TimestampedModel with fields for tracking who created
    and modified records, essential for regulatory compliance.

    Attributes:
        created_by: User who created the record
        updated_by: User who last updated the record
        version: Record version for optimistic locking
    """
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    version: int = 1

    def validate(self) -> None:
        """Validate audited model."""
        super().validate()

        if self.version < 1:
            raise ValueError("Version must be >= 1")

    def increment_version(self, user_id: str) -> None:
        """Increment version and update audit fields.

        Args:
            user_id: ID of user making the change
        """
        self.version += 1
        self.updated_by = user_id
        self.touch()

    @classmethod
    def get_schema(cls) -> Dict[str, str]:
        """Get schema for audited models.

        Returns:
            Dictionary of column names to PostgreSQL types
        """
        schema = super().get_schema()
        schema.update({
            'created_by': 'VARCHAR(255)',
            'updated_by': 'VARCHAR(255)',
            'version': 'INTEGER NOT NULL DEFAULT 1'
        })
        return schema


class ModelValidator:
    """Utility class for common model validations."""

    @staticmethod
    def validate_decimal(
        value: Any,
        field_name: str,
        min_value: Optional[Decimal] = None,
        max_value: Optional[Decimal] = None
    ) -> None:
        """Validate a Decimal field.

        Args:
            value: Value to validate
            field_name: Name of field for error messages
            min_value: Optional minimum value
            max_value: Optional maximum value

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, Decimal):
            raise ValueError(f"{field_name} must be Decimal, got {type(value)}")

        if min_value is not None and value < min_value:
            raise ValueError(f"{field_name} must be >= {min_value}, got {value}")

        if max_value is not None and value > max_value:
            raise ValueError(f"{field_name} must be <= {max_value}, got {value}")

    @staticmethod
    def validate_string(
        value: Any,
        field_name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None
    ) -> None:
        """Validate a string field.

        Args:
            value: Value to validate
            field_name: Name of field for error messages
            min_length: Optional minimum length
            max_length: Optional maximum length
            pattern: Optional regex pattern

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be string, got {type(value)}")

        if min_length is not None and len(value) < min_length:
            raise ValueError(f"{field_name} must be >= {min_length} chars, got {len(value)}")

        if max_length is not None and len(value) > max_length:
            raise ValueError(f"{field_name} must be <= {max_length} chars, got {len(value)}")

        if pattern:
            import re
            if not re.match(pattern, value):
                raise ValueError(f"{field_name} does not match pattern {pattern}")

    @staticmethod
    def validate_datetime(
        value: Any,
        field_name: str,
        require_utc: bool = True
    ) -> None:
        """Validate a datetime field.

        Args:
            value: Value to validate
            field_name: Name of field for error messages
            require_utc: Whether to require UTC timezone

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, datetime):
            raise ValueError(f"{field_name} must be datetime, got {type(value)}")

        if require_utc:
            if not isinstance(value.tzinfo, type(timezone.utc)):
                if value.tzinfo is None:
                    raise ValueError(f"{field_name} must have UTC timezone")

    @staticmethod
    def validate_enum(
        value: Any,
        field_name: str,
        enum_class: Type[Enum]
    ) -> None:
        """Validate an enum field.

        Args:
            value: Value to validate
            field_name: Name of field for error messages
            enum_class: Enum class to validate against

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, enum_class):
            valid_values = [e.value for e in enum_class]
            raise ValueError(
                f"{field_name} must be one of {valid_values}, got {value}"
            )


class ModelRegistry:
    """Registry for tracking all database models.

    Maintains a central registry of all models for schema generation
    and migration purposes.
    """

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, model_class: Type[BaseModel]) -> None:
        """Register a model class.

        Args:
            model_class: Model class to register
        """
        table_name = model_class.get_table_name()
        if table_name in cls._models:
            logger.warning(
                "Model already registered",
                table_name=table_name,
                existing_class=cls._models[table_name].__name__,
                new_class=model_class.__name__
            )
        cls._models[table_name] = model_class
        logger.debug("Model registered", table_name=table_name, model=model_class.__name__)

    @classmethod
    def get_model(cls, table_name: str) -> Optional[Type[BaseModel]]:
        """Get model class by table name.

        Args:
            table_name: Name of database table

        Returns:
            Model class if found, None otherwise
        """
        return cls._models.get(table_name)

    @classmethod
    def get_all_models(cls) -> Dict[str, Type[BaseModel]]:
        """Get all registered models.

        Returns:
            Dictionary mapping table names to model classes
        """
        return cls._models.copy()

    @classmethod
    def generate_schema_sql(cls) -> List[str]:
        """Generate CREATE TABLE statements for all models.

        Returns:
            List of SQL statements
        """
        sql_statements = []

        for table_name, model_class in cls._models.items():
            schema = model_class.get_schema()

            columns = []
            for col_name, col_type in schema.items():
                columns.append(f"    {col_name} {col_type}")

            create_table = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
            create_table += ',\n'.join(columns)
            create_table += "\n);"

            sql_statements.append(create_table)

            logger.debug("Generated schema SQL", table=table_name)

        return sql_statements


def register_model(model_class: Type[BaseModel]) -> Type[BaseModel]:
    """Decorator to automatically register a model class.

    Args:
        model_class: Model class to register

    Returns:
        Same model class (for use as decorator)

    Example:
        @register_model
        @dataclass
        class MyModel(BaseModel):
            ...
    """
    ModelRegistry.register(model_class)
    return model_class
