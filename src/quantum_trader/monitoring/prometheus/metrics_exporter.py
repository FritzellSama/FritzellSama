"""Prometheus metrics exporter for exposing metrics via HTTP.

This module provides an HTTP server that exposes Prometheus metrics for
scraping by Prometheus servers.
"""

import asyncio
import os
from decimal import Decimal
from typing import Dict, Any, Optional
from aiohttp import web
from prometheus_client import (
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
)
from structlog import get_logger

logger = get_logger(__name__)


class MetricsExporter:
    """HTTP server for exporting Prometheus metrics.

    Exposes a /metrics endpoint that can be scraped by Prometheus servers.

    Attributes:
        config: Configuration dictionary
        registry: Prometheus collector registry
        port: HTTP server port
        host: HTTP server host
        app: aiohttp web application
        runner: aiohttp web runner
        site: aiohttp web site

    Example:
        >>> config = {
        ...     "exporter_port": 9090,
        ...     "exporter_host": "0.0.0.0"
        ... }
        >>> registry = CollectorRegistry()
        >>> exporter = MetricsExporter(config, registry)
        >>> await exporter.start()
        >>> # ... metrics are now exposed at http://0.0.0.0:9090/metrics ...
        >>> await exporter.stop()
    """

    def __init__(self, config: Dict[str, Any], registry: CollectorRegistry) -> None:
        """Initialize metrics exporter.

        Args:
            config: Configuration dictionary containing:
                - exporter_port: Port to expose metrics on
                - exporter_host: Host to bind to
                - exporter_path: Path for metrics endpoint
            registry: Prometheus collector registry

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self.registry = registry
        self._validate_config()

        self.port = int(
            self.config.get("exporter_port", os.getenv("METRICS_EXPORTER_PORT", "9090"))
        )
        self.host = self.config.get(
            "exporter_host", os.getenv("METRICS_EXPORTER_HOST", "0.0.0.0")
        )
        self.path = self.config.get(
            "exporter_path", os.getenv("METRICS_EXPORTER_PATH", "/metrics")
        )

        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        self._request_count = 0
        self._error_count = 0

        logger.info(
            "MetricsExporter initialized",
            host=self.host,
            port=self.port,
            path=self.path
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if not isinstance(self.registry, CollectorRegistry):
            raise ValueError("Registry must be a CollectorRegistry instance")

        logger.debug("Config validation passed")

    async def start(self) -> None:
        """Start the metrics exporter HTTP server.

        Raises:
            RuntimeError: If server is already running
        """
        if self.app is not None:
            raise RuntimeError("MetricsExporter is already running")

        try:
            # Create web application
            self.app = web.Application()
            self.app.router.add_get(self.path, self._handle_metrics)
            self.app.router.add_get("/health", self._handle_health)
            self.app.router.add_get("/", self._handle_root)

            # Setup and start server
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            self.site = web.TCPSite(
                self.runner,
                self.host,
                self.port,
                shutdown_timeout=float(
                    self.config.get("shutdown_timeout", os.getenv("EXPORTER_SHUTDOWN_TIMEOUT", "10.0"))
                )
            )
            await self.site.start()

            logger.info(
                "MetricsExporter started",
                url=f"http://{self.host}:{self.port}{self.path}"
            )

        except Exception as e:
            logger.error("Failed to start MetricsExporter", error=str(e), exc_info=True)
            await self._cleanup()
            raise

    async def stop(self) -> None:
        """Stop the metrics exporter HTTP server."""
        if self.app is None:
            logger.warning("MetricsExporter is not running")
            return

        await self._cleanup()

        logger.info(
            "MetricsExporter stopped",
            total_requests=self._request_count,
            total_errors=self._error_count
        )

    async def _cleanup(self) -> None:
        """Clean up server resources."""
        try:
            if self.site:
                await self.site.stop()
                self.site = None

            if self.runner:
                await self.runner.cleanup()
                self.runner = None

            self.app = None

        except Exception as e:
            logger.error("Error during cleanup", error=str(e), exc_info=True)

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """Handle metrics endpoint request.

        Args:
            request: aiohttp request object

        Returns:
            HTTP response with Prometheus metrics
        """
        try:
            self._request_count += 1

            # Generate metrics in Prometheus format
            metrics_data = generate_latest(self.registry)

            logger.debug(
                "Metrics request handled",
                remote=request.remote,
                request_count=self._request_count
            )

            return web.Response(
                body=metrics_data,
                content_type=CONTENT_TYPE_LATEST,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                }
            )

        except Exception as e:
            self._error_count += 1
            logger.error(
                "Error handling metrics request",
                error=str(e),
                remote=request.remote,
                exc_info=True
            )
            return web.Response(
                text=f"Error generating metrics: {str(e)}",
                status=500
            )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check endpoint.

        Args:
            request: aiohttp request object

        Returns:
            HTTP response with health status
        """
        try:
            health_status = {
                "status": "healthy",
                "requests_served": self._request_count,
                "errors": self._error_count,
            }

            return web.json_response(health_status)

        except Exception as e:
            logger.error(
                "Error handling health request",
                error=str(e),
                remote=request.remote,
                exc_info=True
            )
            return web.json_response(
                {"status": "unhealthy", "error": str(e)},
                status=500
            )

    async def _handle_root(self, request: web.Request) -> web.Response:
        """Handle root endpoint request.

        Args:
            request: aiohttp request object

        Returns:
            HTTP response with exporter information
        """
        html = f"""
        <html>
        <head><title>Quantum Trader AI Metrics Exporter</title></head>
        <body>
        <h1>Quantum Trader AI Metrics Exporter</h1>
        <p>
        <a href="{self.path}">Metrics</a><br/>
        <a href="/health">Health</a>
        </p>
        <h2>Statistics</h2>
        <ul>
        <li>Requests served: {self._request_count}</li>
        <li>Errors: {self._error_count}</li>
        </ul>
        </body>
        </html>
        """

        return web.Response(
            text=html,
            content_type="text/html"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get exporter statistics.

        Returns:
            Dictionary with exporter stats
        """
        return {
            "is_running": self.app is not None,
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "url": f"http://{self.host}:{self.port}{self.path}" if self.app else None,
        }


async def create_exporter(
    config: Dict[str, Any],
    registry: CollectorRegistry
) -> MetricsExporter:
    """Create and start a metrics exporter.

    Args:
        config: Configuration dictionary
        registry: Prometheus collector registry

    Returns:
        Started MetricsExporter instance

    Example:
        >>> config = {"exporter_port": 9090}
        >>> registry = CollectorRegistry()
        >>> exporter = await create_exporter(config, registry)
    """
    exporter = MetricsExporter(config, registry)
    await exporter.start()
    return exporter
