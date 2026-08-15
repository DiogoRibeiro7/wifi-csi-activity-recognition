"""Reference-signal and invariant validation for preprocessing.

The per-module tests check that transforms run and preserve shape. A
signal-processing bug survives that easily: a low-pass filter applied along
the wrong axis still returns the right shape, and so does one that attenuates
nothing at all.

These tests assert what each transform is *for* -- using signals whose correct
output is known analytically, and properties that must hold for any input.
"""

from __future__ import annotations

import numpy as np
import pytest

from wifi_activity_recognition.hardware.base import CSIData
from wifi_activity_recognition.preprocessing import (
    butterworth_filter,
    detect_outliers_zscore,
    kalman_filter,
    min_max_normalize,
    moving_average_filter,
    phase_unwrap,
    remove_dc_offset,
    remove_outliers_zscore,
    segment_windows,
    z_score_normalize,
)

FS = 100.0
N_PACKETS = 256


def _packet(amplitude: np.ndarray, phase: np.ndarray | None = None, t: float = 0.0):
    """Build a CSIData packet from an ``(rx, tx, subcarrier)`` array."""
    n_rx, n_tx, n_sc = amplitude.shape
    return CSIData(
        timestamp=t,
        amplitude=amplitude,
        phase=np.zeros_like(amplitude) if phase is None else phase,
        frequency=5.0,
        bandwidth=20.0,
        n_tx=n_tx,
        n_rx=n_rx,
        n_subcarriers=n_sc,
    )


def _sequence_from(signal: np.ndarray) -> list[CSIData]:
    """Turn a 1-D time series into a packet sequence, time on the packet axis."""
    return [
        _packet(np.full((1, 1, 3), value), t=index / FS)
        for index, value in enumerate(signal)
    ]


def _power_at(signal: np.ndarray, freq: float) -> float:
    """Magnitude of the DFT bin nearest ``freq``."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / FS)
    return float(spectrum[np.argmin(np.abs(freqs - freq))])


def _series_of(packets, rx=0, tx=0, sc=0) -> np.ndarray:
    """Extract one subcarrier's amplitude across a packet sequence."""
    return np.array([packet.amplitude[rx, tx, sc] for packet in packets])


# ---------------------------------------------------------------------------
# Temporal filtering against a known reference signal
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_lowpass_attenuates_high_frequencies_and_keeps_low_ones() -> None:
    """The defining property of a low-pass filter, checked in the spectrum.

    A shape-only assertion passes even if the filter runs along the subcarrier
    axis, where a cutoff expressed in Hz has no physical meaning.
    """
    t = np.arange(N_PACKETS) / FS
    keep, remove = np.sin(2 * np.pi * 2 * t), np.sin(2 * np.pi * 40 * t)

    filtered = butterworth_filter(_sequence_from(keep + remove), cutoff=10.0, fs=FS)
    result = _series_of(filtered)

    passband_before, passband_after = _power_at(keep + remove, 2), _power_at(result, 2)
    stopband_before, stopband_after = (
        _power_at(keep + remove, 40),
        _power_at(result, 40),
    )

    assert stopband_after < stopband_before / 100, (
        f"40 Hz component barely attenuated: {stopband_before:.2f} -> "
        f"{stopband_after:.2f} with a 10 Hz cutoff"
    )
    assert passband_after > passband_before * 0.8, (
        f"2 Hz component should survive a 10 Hz cutoff: {passband_before:.2f} -> "
        f"{passband_after:.2f}"
    )


@pytest.mark.regression
def test_filtering_a_sequence_returns_one_packet_per_input() -> None:
    """Temporal filtering must preserve the sequence, not collapse it."""
    packets = _sequence_from(np.random.default_rng(0).normal(size=32))
    filtered = butterworth_filter(packets, cutoff=10.0, fs=FS)

    assert isinstance(filtered, list)
    assert len(filtered) == len(packets)
    for original, result in zip(packets, filtered):
        assert result.amplitude.shape == original.amplitude.shape
        # Metadata must survive filtering.
        assert result.timestamp == original.timestamp
        assert result.n_subcarriers == original.n_subcarriers


def test_constant_signal_is_a_fixed_point_of_the_lowpass() -> None:
    """DC must pass a low-pass filter unchanged once settled."""
    packets = _sequence_from(np.full(N_PACKETS, 3.0))
    settled = _series_of(butterworth_filter(packets, cutoff=10.0, fs=FS))[-32:]

    assert np.allclose(settled, 3.0, atol=1e-6)


@pytest.mark.parametrize("bad_fs", [0.0, -1.0])
def test_butterworth_rejects_invalid_sampling_rate(bad_fs: float) -> None:
    """A non-positive rate has no Nyquist frequency."""
    with pytest.raises(ValueError, match="fs must be positive"):
        butterworth_filter(_sequence_from(np.zeros(8)), cutoff=1.0, fs=bad_fs)


def test_butterworth_rejects_cutoff_above_nyquist() -> None:
    """Cutoff at or above Nyquist is not a representable filter."""
    with pytest.raises(ValueError, match="Nyquist"):
        butterworth_filter(_sequence_from(np.zeros(8)), cutoff=60.0, fs=FS)


def test_filtering_an_empty_sequence_fails_loudly() -> None:
    """An empty sequence must raise rather than return an empty result."""
    with pytest.raises(ValueError, match="empty sequence"):
        butterworth_filter([], cutoff=10.0, fs=FS)


def test_mismatched_packet_shapes_fail_loudly() -> None:
    """Packets of differing shape cannot be stacked along time."""
    packets = [_packet(np.ones((1, 1, 3))), _packet(np.ones((1, 1, 5)))]
    with pytest.raises(ValueError, match="share a shape"):
        butterworth_filter(packets, cutoff=10.0, fs=FS)


# ---------------------------------------------------------------------------
# Smoothing filters reduce noise -- the reason they exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filter_name", ["moving_average", "kalman"])
def test_smoothing_reduces_error_against_the_clean_signal(filter_name: str) -> None:
    """Smoothing a noisy signal must move it closer to the truth."""
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 4 * np.pi, N_PACKETS))
    noisy = clean + rng.normal(scale=0.3, size=clean.shape)
    packets = _sequence_from(noisy)

    if filter_name == "moving_average":
        filtered = moving_average_filter(packets, window_size=9)
    else:
        filtered = kalman_filter(packets, process_variance=1e-3)

    result = _series_of(filtered)
    assert ((result - clean) ** 2).mean() < (
        (noisy - clean) ** 2
    ).mean(), f"{filter_name} did not reduce error against the clean signal"


def test_moving_average_preserves_the_mean_of_a_constant() -> None:
    """Averaging a constant returns the constant, away from the edges."""
    packets = _sequence_from(np.full(64, 2.5))
    interior = _series_of(moving_average_filter(packets, window_size=5))[5:-5]

    assert np.allclose(interior, 2.5, atol=1e-9)


# ---------------------------------------------------------------------------
# Normalization invariants
# ---------------------------------------------------------------------------


def test_min_max_normalize_maps_onto_the_unit_interval() -> None:
    """Output must span exactly [0, 1] for non-degenerate input."""
    rng = np.random.default_rng(0)
    result = min_max_normalize(_packet(rng.normal(size=(2, 2, 16)) * 5 + 3)).amplitude

    assert np.isclose(result.min(), 0.0, atol=1e-9)
    assert np.isclose(result.max(), 1.0, atol=1e-9)


def test_min_max_normalize_is_monotonic() -> None:
    """Normalization must not reorder samples."""
    rng = np.random.default_rng(1)
    original = rng.normal(size=(1, 1, 32))
    result = min_max_normalize(_packet(original)).amplitude

    assert np.array_equal(np.argsort(original.ravel()), np.argsort(result.ravel()))


def test_z_score_normalize_gives_zero_mean_unit_variance() -> None:
    """The defining property of standardization."""
    rng = np.random.default_rng(2)
    result = z_score_normalize(_packet(rng.normal(size=(2, 2, 64)) * 7 - 4)).amplitude

    assert np.isclose(result.mean(), 0.0, atol=1e-9)
    assert np.isclose(result.std(), 1.0, atol=1e-6)


def test_normalizers_do_not_mutate_their_input() -> None:
    """Transforms must be pure: callers may reuse the original packet."""
    original = np.linspace(1, 10, 16).reshape(1, 1, 16)
    packet = _packet(original.copy())

    min_max_normalize(packet)
    z_score_normalize(packet)

    assert np.array_equal(packet.amplitude, original)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_remove_dc_offset_centres_the_signal() -> None:
    """Removing DC must leave zero mean."""
    packet = _packet(np.linspace(5, 15, 32).reshape(1, 1, 32))
    assert np.isclose(remove_dc_offset(packet).amplitude.mean(), 0.0, atol=1e-9)


def test_phase_unwrap_removes_artificial_two_pi_jumps() -> None:
    """Unwrapping a known wrapped ramp must recover the ramp."""
    ramp = np.linspace(0, 6 * np.pi, 64)
    wrapped = np.angle(np.exp(1j * ramp))  # same phase, wrapped into (-pi, pi]

    packet = _packet(np.ones((1, 1, 64)), phase=wrapped.reshape(1, 1, 64))
    unwrapped = phase_unwrap(packet).phase.ravel()

    # Recovered up to a constant offset, and free of the wrapping jumps.
    assert np.allclose(unwrapped - unwrapped[0], ramp - ramp[0], atol=1e-6)
    assert np.abs(np.diff(unwrapped)).max() < np.pi


# ---------------------------------------------------------------------------
# Outliers and segmentation
# ---------------------------------------------------------------------------


def test_outlier_removal_marks_an_extreme_value_and_spares_the_rest() -> None:
    """Outliers become NaN; clean samples must pass through untouched."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=200)
    data[100] = 50.0

    cleaned = remove_outliers_zscore(data, threshold=3.0)

    assert np.isnan(cleaned[100]), "the injected spike was not flagged"
    untouched = np.delete(np.arange(200), 100)
    assert np.allclose(
        cleaned[untouched], data[untouched]
    ), "outlier removal altered samples that were not outliers"


def test_outlier_detection_is_not_dragged_off_by_the_outlier_itself() -> None:
    """Robust statistics: one huge spike must not mask its own detection.

    A mean/standard-deviation rule inflates the spread so much that the spike
    can fall inside the threshold. Median/MAD does not.
    """
    data = np.concatenate([np.zeros(99), [1000.0]])
    assert detect_outliers_zscore(data, threshold=3.0)[-1]


def test_segment_windows_respects_size_and_overlap() -> None:
    """Window contents must match the source packets exactly."""
    packets = _sequence_from(np.arange(10, dtype=float))
    overlap = 0.5
    windows = segment_windows(packets, window_size=4, overlap=overlap)
    step = int(4 * (1 - overlap)) or 1

    assert all(len(window) == 4 for window in windows)
    for index, window in enumerate(windows):
        start = index * step
        assert [packet.timestamp for packet in window] == [
            packet.timestamp for packet in packets[start : start + 4]
        ]


@pytest.mark.parametrize("bad_overlap", [-0.1, 1.0, 1.5])
def test_segment_windows_rejects_invalid_overlap(bad_overlap: float) -> None:
    """Overlap outside [0, 1) cannot produce a forward-moving stride."""
    packets = _sequence_from(np.arange(10, dtype=float))
    with pytest.raises(ValueError, match="overlap"):
        segment_windows(packets, window_size=4, overlap=bad_overlap)
