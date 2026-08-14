"""Behaviour tests for CSI validation and amplitude normalization.

Both live in ``hardware.base`` and every driver depends on them, but neither
was exercised: ``validate_csi_data`` (lines 319-343) never ran, and of the
three ``normalize_csi_amplitude`` methods only ``minmax`` did.

Validation that silently returns the wrong verdict is worse than no validation,
so these assert each rejection reason individually rather than checking that
some arbitrary bad input is refused.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from wifi_activity_recognition.hardware.base import (
    CSIData,
    normalize_csi_amplitude,
    validate_csi_data,
)


def _valid_packet(**overrides) -> CSIData:
    """A packet that passes validation, before any override is applied."""
    shape = (1, 1, 8)
    defaults = dict(
        timestamp=datetime.now().timestamp(),
        amplitude=np.ones(shape),
        phase=np.zeros(shape),
        frequency=5.0,
        bandwidth=20.0,
        n_tx=1,
        n_rx=1,
        n_subcarriers=8,
    )
    defaults.update(overrides)
    return CSIData(**defaults)


# ---------------------------------------------------------------------------
# validate_csi_data
# ---------------------------------------------------------------------------


def test_a_well_formed_packet_validates() -> None:
    """The baseline must pass, or every rejection test below is vacuous."""
    assert validate_csi_data(_valid_packet())


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_amplitude_is_rejected(bad_value: float) -> None:
    """NaN and infinity in amplitude must fail validation."""
    amplitude = np.ones((1, 1, 8))
    amplitude[0, 0, 3] = bad_value
    assert not validate_csi_data(_valid_packet(amplitude=amplitude))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_non_finite_phase_is_rejected(bad_value: float) -> None:
    """Phase is checked separately from amplitude."""
    phase = np.zeros((1, 1, 8))
    phase[0, 0, 2] = bad_value
    assert not validate_csi_data(_valid_packet(phase=phase))


def test_negative_amplitude_is_rejected() -> None:
    """Amplitude is a magnitude and cannot be negative."""
    amplitude = np.ones((1, 1, 8))
    amplitude[0, 0, 0] = -0.5
    assert not validate_csi_data(_valid_packet(amplitude=amplitude))


@pytest.mark.parametrize("out_of_range", [np.pi + 0.1, -np.pi - 0.1])
def test_phase_outside_the_principal_interval_is_rejected(
    out_of_range: float,
) -> None:
    """Phase must lie within [-pi, pi]."""
    phase = np.zeros((1, 1, 8))
    phase[0, 0, 1] = out_of_range
    assert not validate_csi_data(_valid_packet(phase=phase))


def test_phase_exactly_at_the_boundary_is_accepted() -> None:
    """The interval is closed, so +/-pi must not be rejected."""
    phase = np.zeros((1, 1, 8))
    phase[0, 0, 0] = np.pi
    phase[0, 0, 1] = -np.pi
    assert validate_csi_data(_valid_packet(phase=phase))


@pytest.mark.parametrize("offset_seconds", [86_400 * 2, -86_400 * 2])
def test_timestamps_far_from_now_are_rejected(offset_seconds: float) -> None:
    """Packets more than a day away from now are treated as implausible."""
    stamp = datetime.now().timestamp() + offset_seconds
    assert not validate_csi_data(_valid_packet(timestamp=stamp))


def test_malformed_input_is_reported_rather_than_raised() -> None:
    """Validation returns False for junk instead of propagating an exception."""

    class NotAPacket:
        pass

    assert not validate_csi_data(NotAPacket())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# normalize_csi_amplitude
# ---------------------------------------------------------------------------


def test_minmax_maps_onto_the_unit_interval() -> None:
    """Min-max output spans exactly [0, 1]."""
    result = normalize_csi_amplitude(np.array([2.0, 4.0, 6.0]), method="minmax")
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_minmax_leaves_a_constant_array_untouched() -> None:
    """A zero range would divide by zero; the input is returned instead."""
    constant = np.full(5, 3.0)
    assert np.array_equal(normalize_csi_amplitude(constant, method="minmax"), constant)


def test_zscore_gives_zero_mean_and_unit_variance() -> None:
    """The defining property of standardization."""
    rng = np.random.default_rng(0)
    result = normalize_csi_amplitude(rng.normal(size=256) * 5 + 2, method="zscore")
    assert result.mean() == pytest.approx(0.0, abs=1e-9)
    assert result.std() == pytest.approx(1.0, abs=1e-9)


def test_zscore_of_a_constant_centres_without_dividing() -> None:
    """Zero standard deviation must not produce NaN."""
    result = normalize_csi_amplitude(np.full(5, 7.0), method="zscore")
    assert np.all(result == 0.0)
    assert np.isfinite(result).all()


def test_log_normalize_is_monotonic_and_finite_at_zero() -> None:
    """The epsilon exists so that log(0) does not become -inf."""
    result = normalize_csi_amplitude(np.array([0.0, 1.0, 10.0]), method="log")
    assert np.isfinite(result).all()
    assert result[0] < result[1] < result[2]


def test_unknown_method_is_rejected() -> None:
    """An unsupported method must raise rather than silently pass data through."""
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_csi_amplitude(np.ones(4), method="quantile")
