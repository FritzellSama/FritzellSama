"""Fourier transform features for cyclical pattern detection.

This module implements Fourier transform-based features to detect and extract
cyclical patterns, periodic components, and frequency domain characteristics
from price and volume time series. Essential for identifying market cycles.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import polars as pl
import numpy as np
from structlog import get_logger
from scipy import signal
from scipy.fft import fft, ifft, fftfreq

logger = get_logger(__name__)


class FourierFeaturesError(Exception):
    """Base exception for Fourier features computation."""
    pass


class FourierFeatures:
    """Compute Fourier transform features for time series analysis.

    Extracts frequency domain features using Fast Fourier Transform (FFT)
    to identify dominant cycles, spectral power distribution, and periodic
    patterns in price and volume data.

    Attributes:
        config: Configuration dictionary
        n_components: Number of Fourier components to extract
        window_size: Size of the sliding window for transform
        detrend_method: Method for detrending time series

    Example:
        >>> config = {
        ...     'n_components': 10,
        ...     'window_size': 256,
        ...     'detrend_method': 'linear',
        ...     'min_frequency': 0.01,
        ...     'max_frequency': 0.5
        ... }
        >>> fourier = FourierFeatures(config)
        >>> features = await fourier.compute(price_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Fourier features computer.

        Args:
            config: Configuration dictionary with keys:
                - n_components: Number of Fourier components (default: 10)
                - window_size: Window size for FFT (default: 256)
                - detrend_method: Detrending method ('linear', 'constant', 'none')
                - min_frequency: Minimum frequency to consider
                - max_frequency: Maximum frequency to consider
                - apply_windowing: Apply Hann window to reduce spectral leakage
                - normalize_spectrum: Normalize spectral magnitudes

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.n_components = self.config.get('n_components', 10)
        self.window_size = self.config.get('window_size', 256)
        self.detrend_method = self.config.get('detrend_method', 'linear')
        self.min_frequency = Decimal(str(self.config.get('min_frequency', 0.01)))
        self.max_frequency = Decimal(str(self.config.get('max_frequency', 0.5)))
        self.apply_windowing = self.config.get('apply_windowing', True)
        self.normalize_spectrum = self.config.get('normalize_spectrum', True)

        self._lock = asyncio.Lock()

        logger.info(
            "fourier_features_initialized",
            n_components=self.n_components,
            window_size=self.window_size,
            detrend_method=self.detrend_method
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if 'n_components' in self.config and self.config['n_components'] < 1:
            raise ValueError("n_components must be positive")

        if 'window_size' in self.config:
            ws = self.config['window_size']
            if ws < 2 or (ws & (ws - 1)) != 0:
                raise ValueError("window_size must be a power of 2")

        if 'detrend_method' in self.config:
            method = self.config['detrend_method']
            if method not in ['linear', 'constant', 'none']:
                raise ValueError(f"Invalid detrend_method: {method}")

    async def compute(
        self,
        data: pl.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Compute Fourier features from time series data.

        Args:
            data: Input DataFrame with time series
            columns: Columns to compute features for (default: price columns)

        Returns:
            DataFrame with Fourier features added

        Raises:
            FourierFeaturesError: If computation fails
            ValueError: If input data is invalid

        Example:
            >>> features_df = await fourier.compute(
            ...     price_df,
            ...     columns=['close', 'volume']
            ... )
        """
        try:
            if data.is_empty():
                raise ValueError("Input data cannot be empty")

            if data.height < self.window_size:
                logger.warning(
                    "insufficient_data_for_fft",
                    data_length=data.height,
                    required=self.window_size
                )
                # Return data with null features
                return self._add_null_features(data, columns or ['close'])

            # Determine columns to process
            if columns is None:
                columns = [col for col in ['close', 'high', 'low', 'volume']
                          if col in data.columns]

            if not columns:
                raise ValueError("No valid columns found for Fourier analysis")

            result = data.clone()

            # Compute Fourier features for each column
            for col in columns:
                if col not in result.columns:
                    logger.warning(f"column_not_found", column=col)
                    continue

                col_features = await self._compute_column_features(
                    result[col].to_numpy(),
                    col
                )

                # Add features to result
                for feature_name, values in col_features.items():
                    result = result.with_columns([
                        pl.Series(feature_name, values)
                    ])

            logger.info(
                "fourier_features_computed",
                row_count=result.height,
                input_columns=len(columns),
                feature_count=len(result.columns) - len(data.columns)
            )

            return result

        except Exception as e:
            logger.error("fourier_features_computation_failed", error=str(e))
            raise FourierFeaturesError(
                f"Failed to compute Fourier features: {str(e)}"
            ) from e

    async def _compute_column_features(
        self,
        series: np.ndarray,
        col_name: str
    ) -> Dict[str, np.ndarray]:
        """Compute Fourier features for a single column.

        Args:
            series: Time series data as numpy array
            col_name: Name of the column

        Returns:
            Dictionary mapping feature names to value arrays
        """
        features = {}
        n_samples = len(series)

        try:
            # Convert to float for FFT computation
            series_float = series.astype(np.float64)

            # Detrend the series
            if self.detrend_method == 'linear':
                series_detrended = signal.detrend(series_float, type='linear')
            elif self.detrend_method == 'constant':
                series_detrended = signal.detrend(series_float, type='constant')
            else:
                series_detrended = series_float

            # Rolling FFT computation
            for i in range(n_samples):
                if i < self.window_size - 1:
                    # Not enough data yet, use null values
                    for j in range(self.n_components):
                        feature_name = f'{col_name}_fft_magnitude_{j}'
                        if feature_name not in features:
                            features[feature_name] = [Decimal('0.0')] * n_samples
                        features[feature_name][i] = Decimal('0.0')

                    feature_name = f'{col_name}_dominant_frequency'
                    if feature_name not in features:
                        features[feature_name] = [Decimal('0.0')] * n_samples
                    features[feature_name][i] = Decimal('0.0')

                    feature_name = f'{col_name}_spectral_entropy'
                    if feature_name not in features:
                        features[feature_name] = [Decimal('0.0')] * n_samples
                    features[feature_name][i] = Decimal('0.0')

                else:
                    # Extract window
                    window_data = series_detrended[i - self.window_size + 1:i + 1]

                    # Apply windowing function
                    if self.apply_windowing:
                        window_data = window_data * signal.windows.hann(self.window_size)

                    # Compute FFT
                    fft_result = fft(window_data)
                    frequencies = fftfreq(self.window_size)

                    # Get magnitudes (positive frequencies only)
                    n_positive = self.window_size // 2
                    magnitudes = np.abs(fft_result[:n_positive])
                    freq_positive = frequencies[:n_positive]

                    # Normalize if requested
                    if self.normalize_spectrum and magnitudes.sum() > 0:
                        magnitudes = magnitudes / magnitudes.sum()

                    # Filter by frequency range
                    min_freq = float(self.min_frequency)
                    max_freq = float(self.max_frequency)
                    freq_mask = (freq_positive >= min_freq) & (freq_positive <= max_freq)
                    filtered_magnitudes = magnitudes[freq_mask]
                    filtered_frequencies = freq_positive[freq_mask]

                    # Extract top N components
                    if len(filtered_magnitudes) > 0:
                        # Get indices of top components
                        n_extract = min(self.n_components, len(filtered_magnitudes))
                        top_indices = np.argsort(filtered_magnitudes)[-n_extract:][::-1]

                        for j in range(self.n_components):
                            feature_name = f'{col_name}_fft_magnitude_{j}'
                            if feature_name not in features:
                                features[feature_name] = [Decimal('0.0')] * n_samples

                            if j < len(top_indices):
                                magnitude = filtered_magnitudes[top_indices[j]]
                                features[feature_name][i] = Decimal(str(magnitude))
                            else:
                                features[feature_name][i] = Decimal('0.0')

                        # Dominant frequency
                        feature_name = f'{col_name}_dominant_frequency'
                        if feature_name not in features:
                            features[feature_name] = [Decimal('0.0')] * n_samples

                        dominant_idx = top_indices[0]
                        dominant_freq = filtered_frequencies[dominant_idx]
                        features[feature_name][i] = Decimal(str(dominant_freq))

                        # Spectral entropy
                        feature_name = f'{col_name}_spectral_entropy'
                        if feature_name not in features:
                            features[feature_name] = [Decimal('0.0')] * n_samples

                        entropy = self._compute_spectral_entropy(filtered_magnitudes)
                        features[feature_name][i] = Decimal(str(entropy))

                    else:
                        # No valid frequencies
                        for j in range(self.n_components):
                            feature_name = f'{col_name}_fft_magnitude_{j}'
                            if feature_name not in features:
                                features[feature_name] = [Decimal('0.0')] * n_samples
                            features[feature_name][i] = Decimal('0.0')

                        for feature_name in [f'{col_name}_dominant_frequency',
                                           f'{col_name}_spectral_entropy']:
                            if feature_name not in features:
                                features[feature_name] = [Decimal('0.0')] * n_samples
                            features[feature_name][i] = Decimal('0.0')

            logger.debug(
                "column_fourier_features_computed",
                column=col_name,
                feature_count=len(features)
            )

        except Exception as e:
            logger.error(
                "column_fourier_computation_failed",
                column=col_name,
                error=str(e)
            )
            # Return zero features on error
            for j in range(self.n_components):
                features[f'{col_name}_fft_magnitude_{j}'] = [Decimal('0.0')] * n_samples

            features[f'{col_name}_dominant_frequency'] = [Decimal('0.0')] * n_samples
            features[f'{col_name}_spectral_entropy'] = [Decimal('0.0')] * n_samples

        return features

    async def compute_power_spectrum(
        self,
        data: pl.DataFrame,
        column: str = 'close'
    ) -> pl.DataFrame:
        """Compute power spectral density.

        Args:
            data: Input DataFrame
            column: Column to analyze

        Returns:
            DataFrame with frequency and power columns

        Example:
            >>> spectrum = await fourier.compute_power_spectrum(df, 'close')
        """
        try:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in data")

            series = data[column].to_numpy().astype(np.float64)

            # Detrend
            if self.detrend_method != 'none':
                series = signal.detrend(series, type=self.detrend_method)

            # Apply window
            if self.apply_windowing:
                series = series * signal.windows.hann(len(series))

            # Compute power spectral density
            frequencies, psd = signal.periodogram(series, scaling='density')

            # Create result DataFrame
            result = pl.DataFrame({
                'frequency': [Decimal(str(f)) for f in frequencies],
                'power': [Decimal(str(p)) for p in psd]
            })

            # Filter by frequency range
            result = result.filter(
                (pl.col('frequency') >= self.min_frequency) &
                (pl.col('frequency') <= self.max_frequency)
            )

            logger.info(
                "power_spectrum_computed",
                column=column,
                n_frequencies=result.height
            )

            return result

        except Exception as e:
            logger.error("power_spectrum_computation_failed", error=str(e))
            raise FourierFeaturesError(
                f"Failed to compute power spectrum: {str(e)}"
            ) from e

    async def detect_cycles(
        self,
        data: pl.DataFrame,
        column: str = 'close',
        n_cycles: int = 3
    ) -> List[Dict[str, Any]]:
        """Detect dominant cycles in time series.

        Args:
            data: Input DataFrame
            column: Column to analyze
            n_cycles: Number of top cycles to return

        Returns:
            List of cycle information dictionaries

        Example:
            >>> cycles = await fourier.detect_cycles(df, 'close', n_cycles=5)
            >>> for cycle in cycles:
            ...     print(f"Period: {cycle['period']}, Strength: {cycle['strength']}")
        """
        try:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in data")

            series = data[column].to_numpy().astype(np.float64)

            # Detrend
            if self.detrend_method != 'none':
                series = signal.detrend(series, type=self.detrend_method)

            # Apply window
            if self.apply_windowing:
                series = series * signal.windows.hann(len(series))

            # Compute FFT
            fft_result = fft(series)
            frequencies = fftfreq(len(series))

            # Get positive frequencies
            n_positive = len(series) // 2
            magnitudes = np.abs(fft_result[:n_positive])
            freq_positive = frequencies[:n_positive]

            # Filter by frequency range
            min_freq = float(self.min_frequency)
            max_freq = float(self.max_frequency)
            freq_mask = (freq_positive >= min_freq) & (freq_positive <= max_freq)

            filtered_magnitudes = magnitudes[freq_mask]
            filtered_frequencies = freq_positive[freq_mask]

            # Find peaks (dominant cycles)
            peak_indices, peak_properties = signal.find_peaks(
                filtered_magnitudes,
                prominence=filtered_magnitudes.max() * 0.1
            )

            # Sort by magnitude
            if len(peak_indices) > 0:
                peak_magnitudes = filtered_magnitudes[peak_indices]
                sorted_indices = np.argsort(peak_magnitudes)[-n_cycles:][::-1]

                cycles = []
                for idx in sorted_indices:
                    peak_idx = peak_indices[idx]
                    frequency = filtered_frequencies[peak_idx]
                    magnitude = filtered_magnitudes[peak_idx]

                    # Convert frequency to period
                    period = 1.0 / frequency if frequency > 0 else 0.0

                    cycles.append({
                        'frequency': Decimal(str(frequency)),
                        'period': Decimal(str(period)),
                        'strength': Decimal(str(magnitude)),
                        'phase': Decimal(str(np.angle(fft_result[peak_idx])))
                    })

                logger.info(
                    "cycles_detected",
                    column=column,
                    n_cycles=len(cycles)
                )

                return cycles
            else:
                logger.warning("no_cycles_detected", column=column)
                return []

        except Exception as e:
            logger.error("cycle_detection_failed", error=str(e))
            raise FourierFeaturesError(
                f"Failed to detect cycles: {str(e)}"
            ) from e

    async def reconstruct_signal(
        self,
        data: pl.DataFrame,
        column: str,
        n_components: int = 10
    ) -> pl.DataFrame:
        """Reconstruct signal using top N Fourier components.

        Args:
            data: Input DataFrame
            column: Column to reconstruct
            n_components: Number of components to use

        Returns:
            DataFrame with original and reconstructed signals

        Example:
            >>> reconstructed = await fourier.reconstruct_signal(
            ...     df, 'close', n_components=20
            ... )
        """
        try:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in data")

            series = data[column].to_numpy().astype(np.float64)
            original_series = series.copy()

            # Detrend
            trend = None
            if self.detrend_method == 'linear':
                trend = np.polyfit(np.arange(len(series)), series, 1)
                series = signal.detrend(series, type='linear')
            elif self.detrend_method == 'constant':
                trend = series.mean()
                series = signal.detrend(series, type='constant')

            # Compute FFT
            fft_result = fft(series)

            # Keep only top N components
            magnitudes = np.abs(fft_result)
            top_indices = np.argsort(magnitudes)[-n_components:]

            # Zero out other components
            filtered_fft = np.zeros_like(fft_result)
            filtered_fft[top_indices] = fft_result[top_indices]

            # Inverse FFT
            reconstructed = np.real(ifft(filtered_fft))

            # Add trend back
            if trend is not None:
                if isinstance(trend, np.ndarray):
                    reconstructed += np.polyval(trend, np.arange(len(reconstructed)))
                else:
                    reconstructed += trend

            # Create result DataFrame
            result = data.select(['timestamp'] if 'timestamp' in data.columns else [])
            result = result.with_columns([
                pl.Series(f'{column}_original', [Decimal(str(v)) for v in original_series]),
                pl.Series(f'{column}_reconstructed', [Decimal(str(v)) for v in reconstructed]),
                pl.Series(f'{column}_residual', [
                    Decimal(str(original_series[i] - reconstructed[i]))
                    for i in range(len(original_series))
                ])
            ])

            logger.info(
                "signal_reconstructed",
                column=column,
                n_components=n_components
            )

            return result

        except Exception as e:
            logger.error("signal_reconstruction_failed", error=str(e))
            raise FourierFeaturesError(
                f"Failed to reconstruct signal: {str(e)}"
            ) from e

    @staticmethod
    def _compute_spectral_entropy(magnitudes: np.ndarray) -> float:
        """Compute spectral entropy from magnitude spectrum.

        Args:
            magnitudes: Magnitude spectrum

        Returns:
            Spectral entropy value
        """
        # Normalize to probability distribution
        if magnitudes.sum() == 0:
            return 0.0

        prob_dist = magnitudes / magnitudes.sum()

        # Compute entropy
        entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))

        return float(entropy)

    def _add_null_features(
        self,
        data: pl.DataFrame,
        columns: List[str]
    ) -> pl.DataFrame:
        """Add null Fourier features when data is insufficient.

        Args:
            data: Input DataFrame
            columns: Columns to add features for

        Returns:
            DataFrame with null features
        """
        result = data

        for col in columns:
            for j in range(self.n_components):
                result = result.with_columns([
                    pl.lit(Decimal('0.0')).alias(f'{col}_fft_magnitude_{j}')
                ])

            result = result.with_columns([
                pl.lit(Decimal('0.0')).alias(f'{col}_dominant_frequency'),
                pl.lit(Decimal('0.0')).alias(f'{col}_spectral_entropy')
            ])

        return result
