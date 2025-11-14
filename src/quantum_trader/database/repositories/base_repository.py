"""Base repository pattern implementation for database operations.

Provides generic CRUD operations with async support, connection pooling,
and query optimization for all database models.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Type, TypeVar, Generic
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from structlog import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class RepositoryError(Exception):
    """Base exception for repository errors."""
    pass


class NotFoundError(RepositoryError):
    """Entity not found error."""
    pass


class DuplicateError(RepositoryError):
    """Duplicate entity error."""
    pass


class BaseRepository(ABC, Generic[T]):
    """Generic base repository for database operations.

    Implements the repository pattern with async operations,
    transaction support, and connection pooling.

    Attributes:
        db_pool: AsyncPG connection pool
        model_class: Class of model this repository manages
        table_name: Database table name
    """

    def __init__(
        self,
        db_pool: Any,
        model_class: Type[T],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize repository.

        Args:
            db_pool: AsyncPG connection pool
            model_class: Model class for this repository
            config: Optional configuration dictionary
        """
        self.db_pool = db_pool
        self.model_class = model_class
        self.config = config or {}

        # Get table name from model
        if hasattr(model_class, 'get_table_name'):
            self.table_name = model_class.get_table_name()
        else:
            # Fallback to class name in snake_case
            self.table_name = self._to_snake_case(model_class.__name__)

        self.query_timeout = self.config.get('query_timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay_ms', 100) / 1000.0

        logger.info(
            "Repository initialized",
            table=self.table_name,
            model=model_class.__name__
        )

    def _to_snake_case(self, name: str) -> str:
        """Convert CamelCase to snake_case.

        Args:
            name: CamelCase string

        Returns:
            snake_case string
        """
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append('_')
            result.append(char.lower())
        return ''.join(result)

    async def create(self, entity: T) -> T:
        """Create a new entity in the database.

        Args:
            entity: Entity to create

        Returns:
            Created entity with populated ID

        Raises:
            DuplicateError: If entity already exists
            RepositoryError: If creation fails
        """
        try:
            entity_dict = self._entity_to_dict(entity)
            columns = list(entity_dict.keys())
            values = list(entity_dict.values())

            # Build parameterized query
            placeholders = ', '.join(f'${i + 1}' for i in range(len(columns)))
            cols_str = ', '.join(columns)

            query = f"""
                INSERT INTO {self.table_name} ({cols_str})
                VALUES ({placeholders})
                RETURNING *
            """

            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, *values, timeout=self.query_timeout)

            created_entity = self._row_to_entity(row)

            logger.info(
                "Entity created",
                table=self.table_name,
                id=getattr(created_entity, 'id', None)
            )

            return created_entity

        except Exception as e:
            if 'unique constraint' in str(e).lower():
                raise DuplicateError(f"Entity already exists: {e}")

            logger.error("Entity creation failed", table=self.table_name, error=str(e))
            raise RepositoryError(f"Create failed: {e}")

    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity if found, None otherwise

        Raises:
            RepositoryError: If query fails
        """
        try:
            query = f"SELECT * FROM {self.table_name} WHERE id = $1"

            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, entity_id, timeout=self.query_timeout)

            if row:
                entity = self._row_to_entity(row)
                logger.debug("Entity retrieved", table=self.table_name, id=entity_id)
                return entity

            return None

        except Exception as e:
            logger.error("Get by ID failed", table=self.table_name, id=entity_id, error=str(e))
            raise RepositoryError(f"Get by ID failed: {e}")

    async def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[T]:
        """Get all entities with optional pagination.

        Args:
            limit: Maximum number of entities to return
            offset: Number of entities to skip
            order_by: Column to order by (e.g., 'created_at DESC')

        Returns:
            List of entities

        Raises:
            RepositoryError: If query fails
        """
        try:
            query = f"SELECT * FROM {self.table_name}"

            if order_by:
                query += f" ORDER BY {order_by}"
            else:
                query += " ORDER BY created_at DESC"

            params = []
            if limit:
                params.append(limit)
                query += f" LIMIT ${len(params)}"

            if offset:
                params.append(offset)
                query += f" OFFSET ${len(params)}"

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params, timeout=self.query_timeout)

            entities = [self._row_to_entity(row) for row in rows]

            logger.debug("Entities retrieved", table=self.table_name, count=len(entities))
            return entities

        except Exception as e:
            logger.error("Get all failed", table=self.table_name, error=str(e))
            raise RepositoryError(f"Get all failed: {e}")

    async def update(self, entity: T) -> T:
        """Update an existing entity.

        Args:
            entity: Entity to update (must have ID)

        Returns:
            Updated entity

        Raises:
            NotFoundError: If entity not found
            RepositoryError: If update fails
        """
        entity_id = getattr(entity, 'id', None)
        if not entity_id:
            raise RepositoryError("Entity must have ID to update")

        try:
            # Update timestamp if model supports it
            if hasattr(entity, 'touch'):
                entity.touch()

            entity_dict = self._entity_to_dict(entity)
            # Remove ID from update dict
            entity_dict.pop('id', None)

            columns = list(entity_dict.keys())
            values = list(entity_dict.values())
            values.append(entity_id)  # Add ID as last parameter

            # Build SET clause
            set_clause = ', '.join(f"{col} = ${i + 1}" for i, col in enumerate(columns))

            query = f"""
                UPDATE {self.table_name}
                SET {set_clause}
                WHERE id = ${len(values)}
                RETURNING *
            """

            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, *values, timeout=self.query_timeout)

            if not row:
                raise NotFoundError(f"Entity not found: {entity_id}")

            updated_entity = self._row_to_entity(row)

            logger.info("Entity updated", table=self.table_name, id=entity_id)
            return updated_entity

        except NotFoundError:
            raise
        except Exception as e:
            logger.error("Update failed", table=self.table_name, id=entity_id, error=str(e))
            raise RepositoryError(f"Update failed: {e}")

    async def delete(self, entity_id: str, soft: bool = False) -> bool:
        """Delete an entity by ID.

        Args:
            entity_id: Entity ID
            soft: If True, soft delete (set deleted_at), otherwise hard delete

        Returns:
            True if deleted, False if not found

        Raises:
            RepositoryError: If deletion fails
        """
        try:
            if soft:
                # Soft delete - set deleted_at timestamp
                query = f"""
                    UPDATE {self.table_name}
                    SET deleted_at = $1, updated_at = $1
                    WHERE id = $2 AND deleted_at IS NULL
                    RETURNING id
                """
                now = datetime.now(timezone.utc)

                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow(query, now, entity_id, timeout=self.query_timeout)

            else:
                # Hard delete - remove from database
                query = f"DELETE FROM {self.table_name} WHERE id = $1 RETURNING id"

                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow(query, entity_id, timeout=self.query_timeout)

            deleted = row is not None

            if deleted:
                logger.info(
                    "Entity deleted",
                    table=self.table_name,
                    id=entity_id,
                    soft=soft
                )

            return deleted

        except Exception as e:
            logger.error("Delete failed", table=self.table_name, id=entity_id, error=str(e))
            raise RepositoryError(f"Delete failed: {e}")

    async def find_by(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[T]:
        """Find entities matching filters.

        Args:
            filters: Dictionary of column: value filters
            limit: Maximum results to return
            order_by: Column to order by

        Returns:
            List of matching entities

        Raises:
            RepositoryError: If query fails
        """
        try:
            if not filters:
                return await self.get_all(limit=limit, order_by=order_by)

            # Build WHERE clause
            where_parts = []
            values = []
            for i, (col, val) in enumerate(filters.items(), 1):
                where_parts.append(f"{col} = ${i}")
                values.append(val)

            where_clause = ' AND '.join(where_parts)
            query = f"SELECT * FROM {self.table_name} WHERE {where_clause}"

            if order_by:
                query += f" ORDER BY {order_by}"

            if limit:
                query += f" LIMIT {limit}"

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *values, timeout=self.query_timeout)

            entities = [self._row_to_entity(row) for row in rows]

            logger.debug(
                "Entities found",
                table=self.table_name,
                filters=filters,
                count=len(entities)
            )

            return entities

        except Exception as e:
            logger.error("Find by failed", table=self.table_name, filters=filters, error=str(e))
            raise RepositoryError(f"Find by failed: {e}")

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching optional filters.

        Args:
            filters: Optional dictionary of column: value filters

        Returns:
            Count of matching entities

        Raises:
            RepositoryError: If query fails
        """
        try:
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            values = []

            if filters:
                where_parts = []
                for i, (col, val) in enumerate(filters.items(), 1):
                    where_parts.append(f"{col} = ${i}")
                    values.append(val)

                where_clause = ' AND '.join(where_parts)
                query += f" WHERE {where_clause}"

            async with self.db_pool.acquire() as conn:
                count = await conn.fetchval(query, *values, timeout=self.query_timeout)

            logger.debug("Count executed", table=self.table_name, filters=filters, count=count)
            return count

        except Exception as e:
            logger.error("Count failed", table=self.table_name, error=str(e))
            raise RepositoryError(f"Count failed: {e}")

    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists by ID.

        Args:
            entity_id: Entity ID

        Returns:
            True if exists, False otherwise
        """
        try:
            query = f"SELECT EXISTS(SELECT 1 FROM {self.table_name} WHERE id = $1)"

            async with self.db_pool.acquire() as conn:
                exists = await conn.fetchval(query, entity_id, timeout=self.query_timeout)

            return bool(exists)

        except Exception as e:
            logger.error("Exists check failed", table=self.table_name, id=entity_id, error=str(e))
            raise RepositoryError(f"Exists check failed: {e}")

    async def bulk_create(self, entities: List[T]) -> List[T]:
        """Create multiple entities in a single transaction.

        Args:
            entities: List of entities to create

        Returns:
            List of created entities with IDs

        Raises:
            RepositoryError: If creation fails
        """
        if not entities:
            return []

        try:
            entity_dicts = [self._entity_to_dict(e) for e in entities]
            columns = list(entity_dicts[0].keys())

            # Build bulk insert query
            placeholders_per_row = len(columns)
            rows_placeholders = []

            for i in range(len(entity_dicts)):
                row_placeholders = ', '.join(
                    f'${i * placeholders_per_row + j + 1}'
                    for j in range(placeholders_per_row)
                )
                rows_placeholders.append(f'({row_placeholders})')

            cols_str = ', '.join(columns)
            values_str = ', '.join(rows_placeholders)

            query = f"""
                INSERT INTO {self.table_name} ({cols_str})
                VALUES {values_str}
                RETURNING *
            """

            # Flatten values list
            values = []
            for entity_dict in entity_dicts:
                values.extend(entity_dict.values())

            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *values, timeout=self.query_timeout)

            created_entities = [self._row_to_entity(row) for row in rows]

            logger.info("Bulk create completed", table=self.table_name, count=len(created_entities))
            return created_entities

        except Exception as e:
            logger.error("Bulk create failed", table=self.table_name, count=len(entities), error=str(e))
            raise RepositoryError(f"Bulk create failed: {e}")

    @abstractmethod
    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """Convert entity to dictionary for database operations.

        Args:
            entity: Entity to convert

        Returns:
            Dictionary with database column values
        """
        pass

    @abstractmethod
    def _row_to_entity(self, row: Any) -> T:
        """Convert database row to entity.

        Args:
            row: Database row

        Returns:
            Entity instance
        """
        pass


class ReadOnlyRepository(BaseRepository[T]):
    """Read-only repository that disables write operations.

    Useful for views or when enforcing read-only access patterns.
    """

    async def create(self, entity: T) -> T:
        """Disabled in read-only repository."""
        raise RepositoryError("Create operation not allowed in read-only repository")

    async def update(self, entity: T) -> T:
        """Disabled in read-only repository."""
        raise RepositoryError("Update operation not allowed in read-only repository")

    async def delete(self, entity_id: str, soft: bool = False) -> bool:
        """Disabled in read-only repository."""
        raise RepositoryError("Delete operation not allowed in read-only repository")

    async def bulk_create(self, entities: List[T]) -> List[T]:
        """Disabled in read-only repository."""
        raise RepositoryError("Bulk create operation not allowed in read-only repository")
