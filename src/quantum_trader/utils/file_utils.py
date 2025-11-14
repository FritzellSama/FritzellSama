"""File Utility Functions.

Production-ready file operations with atomic writes, safe reads,
compression, and backup management.
"""

import os
import shutil
import json
import gzip
from pathlib import Path
from decimal import Decimal
from typing import Optional, Dict, List, Any, Union
from datetime import datetime
import tempfile
from structlog import get_logger

logger = get_logger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object

    Raises:
        OSError: If directory creation fails
    """
    try:
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        logger.debug("directory_ensured", path=str(dir_path))
        return dir_path

    except Exception as e:
        logger.error("directory_creation_failed", path=str(path), error=str(e))
        raise OSError(f"Failed to ensure directory: {e}")


def atomic_write(
    file_path: Union[str, Path],
    content: Union[str, bytes],
    mode: str = 'w',
    encoding: str = 'utf-8'
) -> None:
    """Write file atomically using temp file and rename.

    Args:
        file_path: Target file path
        content: Content to write
        mode: Write mode ('w' or 'wb')
        encoding: Text encoding (for text mode)

    Raises:
        OSError: If write fails
    """
    try:
        path = Path(file_path)

        # Ensure parent directory exists
        ensure_directory(path.parent)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode=mode,
            encoding=encoding if 'b' not in mode else None,
            dir=path.parent,
            delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(content)

        # Atomic rename
        shutil.move(tmp_path, str(path))

        logger.debug("atomic_write_completed", path=str(path), size=len(content))

    except Exception as e:
        logger.error("atomic_write_failed", path=str(file_path), error=str(e))
        # Clean up temp file if exists
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass
        raise OSError(f"Failed to write file atomically: {e}")


def safe_read(
    file_path: Union[str, Path],
    mode: str = 'r',
    encoding: str = 'utf-8',
    default: Optional[Any] = None
) -> Optional[Union[str, bytes]]:
    """Safely read file with error handling.

    Args:
        file_path: File path
        mode: Read mode ('r' or 'rb')
        encoding: Text encoding (for text mode)
        default: Default value if file doesn't exist

    Returns:
        File content or default value
    """
    try:
        path = Path(file_path)

        if not path.exists():
            logger.debug("file_not_found", path=str(path))
            return default

        with open(path, mode=mode, encoding=encoding if 'b' not in mode else None) as f:
            content = f.read()

        logger.debug("file_read", path=str(path), size=len(content))
        return content

    except Exception as e:
        logger.error("file_read_failed", path=str(file_path), error=str(e))
        return default


def read_json(
    file_path: Union[str, Path],
    default: Optional[Dict] = None
) -> Optional[Dict]:
    """Read JSON file.

    Args:
        file_path: JSON file path
        default: Default value if file doesn't exist or invalid

    Returns:
        Parsed JSON or default value
    """
    try:
        content = safe_read(file_path, default='{}')

        if content is None or content == '':
            return default

        data = json.loads(content)
        logger.debug("json_read", path=str(file_path), keys=len(data))
        return data

    except json.JSONDecodeError as e:
        logger.error("json_parse_failed", path=str(file_path), error=str(e))
        return default
    except Exception as e:
        logger.error("json_read_failed", path=str(file_path), error=str(e))
        return default


def write_json(
    file_path: Union[str, Path],
    data: Dict,
    indent: int = 2,
    sort_keys: bool = True
) -> bool:
    """Write JSON file atomically.

    Args:
        file_path: JSON file path
        data: Data to write
        indent: JSON indentation
        sort_keys: Sort dictionary keys

    Returns:
        True if successful
    """
    try:
        # Convert Decimal to string for JSON serialization
        def decimal_default(obj: Any) -> Any:
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json_content = json.dumps(
            data,
            indent=indent,
            sort_keys=sort_keys,
            default=decimal_default
        )

        atomic_write(file_path, json_content, mode='w')

        logger.debug("json_written", path=str(file_path), keys=len(data))
        return True

    except Exception as e:
        logger.error("json_write_failed", path=str(file_path), error=str(e))
        return False


def compress_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    remove_original: bool = False
) -> Optional[Path]:
    """Compress file using gzip.

    Args:
        input_path: Input file path
        output_path: Output path (defaults to input_path.gz)
        remove_original: Remove original after compression

    Returns:
        Output path or None on error
    """
    try:
        in_path = Path(input_path)

        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            out_path = Path(f"{input_path}.gz")
        else:
            out_path = Path(output_path)

        # Compress
        with open(in_path, 'rb') as f_in:
            with gzip.open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove original if requested
        if remove_original:
            in_path.unlink()
            logger.debug("original_removed", path=str(in_path))

        logger.info(
            "file_compressed",
            input=str(in_path),
            output=str(out_path),
            size=out_path.stat().st_size
        )

        return out_path

    except Exception as e:
        logger.error(
            "compression_failed",
            input=str(input_path),
            error=str(e)
        )
        return None


def decompress_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    remove_compressed: bool = False
) -> Optional[Path]:
    """Decompress gzipped file.

    Args:
        input_path: Compressed file path
        output_path: Output path (defaults to input without .gz)
        remove_compressed: Remove compressed file after decompression

    Returns:
        Output path or None on error
    """
    try:
        in_path = Path(input_path)

        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            if str(in_path).endswith('.gz'):
                out_path = Path(str(in_path)[:-3])
            else:
                out_path = Path(f"{input_path}.decompressed")
        else:
            out_path = Path(output_path)

        # Decompress
        with gzip.open(in_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove compressed if requested
        if remove_compressed:
            in_path.unlink()
            logger.debug("compressed_removed", path=str(in_path))

        logger.info(
            "file_decompressed",
            input=str(in_path),
            output=str(out_path),
            size=out_path.stat().st_size
        )

        return out_path

    except Exception as e:
        logger.error(
            "decompression_failed",
            input=str(input_path),
            error=str(e)
        )
        return None


def create_backup(
    file_path: Union[str, Path],
    backup_dir: Optional[Union[str, Path]] = None,
    keep_n_backups: int = 5
) -> Optional[Path]:
    """Create timestamped backup of file.

    Args:
        file_path: File to backup
        backup_dir: Backup directory (defaults to file_path.backups)
        keep_n_backups: Number of backups to keep

    Returns:
        Backup path or None on error
    """
    try:
        path = Path(file_path)

        if not path.exists():
            logger.warning("backup_source_not_found", path=str(path))
            return None

        # Create backup directory
        if backup_dir is None:
            bak_dir = path.parent / f"{path.name}.backups"
        else:
            bak_dir = Path(backup_dir)

        ensure_directory(bak_dir)

        # Create backup with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.name}.{timestamp}.bak"
        backup_path = bak_dir / backup_name

        # Copy file
        shutil.copy2(path, backup_path)

        logger.info("backup_created", original=str(path), backup=str(backup_path))

        # Clean old backups
        _cleanup_old_backups(bak_dir, path.name, keep_n_backups)

        return backup_path

    except Exception as e:
        logger.error("backup_failed", path=str(file_path), error=str(e))
        return None


def _cleanup_old_backups(
    backup_dir: Path,
    base_name: str,
    keep_n: int
) -> None:
    """Remove old backups keeping only the newest N.

    Args:
        backup_dir: Backup directory
        base_name: Base file name
        keep_n: Number to keep
    """
    try:
        # Find all backups
        pattern = f"{base_name}.*.bak"
        backups = sorted(
            backup_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove old ones
        for old_backup in backups[keep_n:]:
            old_backup.unlink()
            logger.debug("old_backup_removed", path=str(old_backup))

    except Exception as e:
        logger.error("backup_cleanup_failed", error=str(e))


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes.

    Args:
        file_path: File path

    Returns:
        File size in bytes (0 if not found)
    """
    try:
        path = Path(file_path)
        if path.exists():
            return path.stat().st_size
        return 0

    except Exception as e:
        logger.error("file_size_failed", path=str(file_path), error=str(e))
        return 0


def get_file_age_seconds(file_path: Union[str, Path]) -> float:
    """Get file age in seconds since last modification.

    Args:
        file_path: File path

    Returns:
        Age in seconds (0 if not found)
    """
    try:
        path = Path(file_path)
        if path.exists():
            import time
            mtime = path.stat().st_mtime
            return time.time() - mtime
        return 0.0

    except Exception as e:
        logger.error("file_age_failed", path=str(file_path), error=str(e))
        return 0.0


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """List files in directory.

    Args:
        directory: Directory path
        pattern: Glob pattern
        recursive: Search recursively

    Returns:
        List of file paths
    """
    try:
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning("directory_not_found", path=str(dir_path))
            return []

        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        # Filter out directories
        files = [f for f in files if f.is_file()]

        logger.debug(
            "files_listed",
            directory=str(dir_path),
            pattern=pattern,
            count=len(files)
        )

        return files

    except Exception as e:
        logger.error("list_files_failed", directory=str(directory), error=str(e))
        return []


def delete_file(file_path: Union[str, Path], missing_ok: bool = True) -> bool:
    """Delete file safely.

    Args:
        file_path: File path
        missing_ok: Don't raise error if file doesn't exist

    Returns:
        True if deleted or missing_ok

    Raises:
        FileNotFoundError: If file doesn't exist and not missing_ok
    """
    try:
        path = Path(file_path)

        if not path.exists():
            if missing_ok:
                return True
            raise FileNotFoundError(f"File not found: {file_path}")

        path.unlink()
        logger.debug("file_deleted", path=str(path))
        return True

    except Exception as e:
        logger.error("file_delete_failed", path=str(file_path), error=str(e))
        if not missing_ok:
            raise
        return False


def copy_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """Copy file with safety checks.

    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Overwrite existing destination

    Returns:
        True if successful

    Raises:
        FileExistsError: If destination exists and not overwrite
    """
    try:
        src_path = Path(source)
        dst_path = Path(destination)

        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {destination}")

        # Ensure destination directory
        ensure_directory(dst_path.parent)

        # Copy file
        shutil.copy2(src_path, dst_path)

        logger.debug("file_copied", source=str(src_path), dest=str(dst_path))
        return True

    except Exception as e:
        logger.error(
            "file_copy_failed",
            source=str(source),
            dest=str(destination),
            error=str(e)
        )
        raise


def move_file(
    source: Union[str, Path],
    destination: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """Move file with safety checks.

    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Overwrite existing destination

    Returns:
        True if successful
    """
    try:
        src_path = Path(source)
        dst_path = Path(destination)

        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {destination}")

        # Ensure destination directory
        ensure_directory(dst_path.parent)

        # Move file
        shutil.move(str(src_path), str(dst_path))

        logger.debug("file_moved", source=str(src_path), dest=str(dst_path))
        return True

    except Exception as e:
        logger.error(
            "file_move_failed",
            source=str(source),
            dest=str(destination),
            error=str(e)
        )
        raise
