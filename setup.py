"""
Quantum Trader AI - Setup Configuration

Institutional-grade automated trading system managing billions in capital.
Production-ready deployment with <10ms latency and 10K+ trades/day.

Stack: Python 3.12+, Polars, AsyncIO, FastAPI, PyTorch 2.0+
"""

import os
import sys
from pathlib import Path
from typing import List

from setuptools import find_packages, setup

# Ensure Python 3.12+
if sys.version_info < (3, 12):
    sys.exit("ERROR: Quantum Trader AI requires Python 3.12 or higher.")

# Read version from file
version_file = Path(__file__).parent / "src" / "quantum_trader" / "__version__.py"
version_info = {}
if version_file.exists():
    with open(version_file) as f:
        exec(f.read(), version_info)
    VERSION = version_info.get("__version__", "0.1.0")
else:
    VERSION = os.getenv("QUANTUM_TRADER_VERSION", "0.1.0")

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

# Core dependencies - production-ready, battle-tested versions
INSTALL_REQUIRES = [
    # Data processing - CRITICAL: Use Polars, NOT pandas
    "polars>=0.20.0,<1.0.0",
    "pyarrow>=14.0.0",  # Required for Polars
    # Async runtime
    "asyncio>=3.4.3",
    "aiohttp>=3.9.0",
    "aiofiles>=23.2.0",
    "asyncpg>=0.29.0",  # Async PostgreSQL
    "redis[hiredis]>=5.0.0",  # Async Redis with hiredis for performance
    # Web framework
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    # Machine Learning
    "torch>=2.0.0,<3.0.0",
    "numpy>=1.24.0,<2.0.0",
    "scikit-learn>=1.4.0",
    # Technical analysis
    "ta-lib>=0.4.28",  # Requires TA-Lib C library
    # Logging and monitoring
    "structlog>=24.1.0",
    "python-json-logger>=2.0.7",
    # Configuration
    "pyyaml>=6.0.1",
    "python-dotenv>=1.0.0",
    # Database
    "sqlalchemy[asyncio]>=2.0.0",
    "alembic>=1.13.0",
    # Validation and serialization
    "marshmallow>=3.20.0",
    "cerberus>=1.3.5",
    # Encryption and security
    "cryptography>=42.0.0",
    "pyjwt>=2.8.0",
    "bcrypt>=4.1.0",
    # HTTP clients
    "httpx>=0.26.0",
    # Date/time utilities
    "python-dateutil>=2.8.2",
    "pytz>=2024.1",
    # Retry logic and resilience
    "tenacity>=8.2.3",
    "backoff>=2.2.1",
    # Rate limiting
    "limits>=3.7.0",
    # Metrics and monitoring
    "prometheus-client>=0.19.0",
    # Process management
    "supervisor>=4.2.5",
    # CLI utilities
    "click>=8.1.7",
    "rich>=13.7.0",  # Beautiful terminal output
    # Data validation
    "jsonschema>=4.21.0",
    # Caching
    "cachetools>=5.3.2",
    # Message queue (optional)
    "aiormq>=6.8.0",  # Async RabbitMQ
]

# Development dependencies
DEV_REQUIRES = [
    # Testing
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "pytest-timeout>=2.2.0",
    "pytest-xdist>=3.5.0",  # Parallel testing
    "hypothesis>=6.98.0",  # Property-based testing
    "faker>=22.0.0",  # Test data generation
    # Code quality
    "black>=24.1.0",
    "isort>=5.13.0",
    "ruff>=0.2.0",
    "pylint>=3.0.0",
    "mypy>=1.8.0",
    "bandit>=1.7.7",  # Security linting
    "safety>=3.0.0",  # Dependency security check
    # Pre-commit hooks
    "pre-commit>=3.6.0",
    # Documentation
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=2.0.0",
    "sphinx-autodoc-typehints>=1.25.0",
    # Performance profiling
    "py-spy>=0.3.14",
    "memray>=1.11.0",
    "scalene>=1.5.38",
    # Debugging
    "ipdb>=0.13.13",
    "ipython>=8.20.0",
    # Load testing
    "locust>=2.20.0",
]

# Exchange-specific dependencies
EXCHANGE_REQUIRES = [
    "ccxt>=4.2.0",  # Unified exchange API
]

# ML/AI additional dependencies
ML_REQUIRES = [
    "tensorflow>=2.15.0,<3.0.0",  # Optional: For TensorFlow models
    "xgboost>=2.0.0",
    "lightgbm>=4.3.0",
    "optuna>=3.5.0",  # Hyperparameter optimization
    "shap>=0.44.0",  # Model interpretation
]

# Monitoring and observability
MONITORING_REQUIRES = [
    "sentry-sdk>=1.40.0",
    "datadog>=0.49.0",
    "opentelemetry-api>=1.22.0",
    "opentelemetry-sdk>=1.22.0",
]

# Deployment dependencies
DEPLOY_REQUIRES = [
    "ansible>=9.0.0",
    "docker>=7.0.0",
    "kubernetes>=29.0.0",
]

# All optional dependencies
EXTRAS_REQUIRE = {
    "dev": DEV_REQUIRES,
    "exchange": EXCHANGE_REQUIRES,
    "ml": ML_REQUIRES,
    "monitoring": MONITORING_REQUIRES,
    "deploy": DEPLOY_REQUIRES,
    "all": DEV_REQUIRES + EXCHANGE_REQUIRES + ML_REQUIRES + MONITORING_REQUIRES + DEPLOY_REQUIRES,
}

setup(
    name="quantum-trader-ai",
    version=VERSION,
    description="Institutional-grade automated cryptocurrency trading system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Quantum Trader AI Team",
    author_email=os.getenv("QUANTUM_TRADER_CONTACT_EMAIL", "team@quantumtrader.ai"),
    url="https://github.com/quantum-trader-ai/quantum-trader-ai",
    project_urls={
        "Documentation": "https://docs.quantumtrader.ai",
        "Source": "https://github.com/quantum-trader-ai/quantum-trader-ai",
        "Tracker": "https://github.com/quantum-trader-ai/quantum-trader-ai/issues",
    },
    license="Proprietary",  # Update as needed
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        # Intended audience
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        # Topics
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # License (update as needed)
        "License :: Other/Proprietary License",
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Implementation :: CPython",
        # Operating systems
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        # Framework
        "Framework :: AsyncIO",
        "Framework :: FastAPI",
        # Typing
        "Typing :: Typed",
    ],
    keywords=[
        "trading",
        "cryptocurrency",
        "bitcoin",
        "ethereum",
        "algorithmic-trading",
        "high-frequency-trading",
        "market-making",
        "quantitative-finance",
        "machine-learning",
        "deep-learning",
        "polars",
        "asyncio",
        "institutional",
    ],
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "quantum_trader": [
            "py.typed",  # PEP 561 marker for type hints
            "config/**/*.yaml",
            "config/**/*.json",
        ],
    },
    python_requires=">=3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "quantum-trader=quantum_trader.cli:main",
            "qt-trade=quantum_trader.cli:trade",
            "qt-backtest=quantum_trader.cli:backtest",
            "qt-analyze=quantum_trader.cli:analyze",
            "qt-monitor=quantum_trader.cli:monitor",
        ],
    },
    zip_safe=False,  # Required for type hints
    # Platform-specific configurations
    platforms=["any"],
    # Test suite
    test_suite="tests",
    tests_require=DEV_REQUIRES,
)
