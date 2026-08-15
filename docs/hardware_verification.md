# Hardware verification

Tracking issue: [#22](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/22).

The value of this package depends on its hardware support, and mock coverage
does not establish that a real Intel 5300 or ESP32 works. This page separates
**what CI proves** from **what only a device can prove**, gives a per-platform
procedure, and provides a table for recording real-device runs.

## Two kinds of coverage, kept apart

| | CI (mock) | Real device |
|---|---|---|
| Runs on | every push and PR | manually, with hardware attached |
| Proves | driver contract, packet structure, factory wiring, pipeline integration | protocol compatibility, firmware behaviour, timing, real failure modes |
| Tool | `pytest tests/hardware/` | `scripts/verify_device.py` |
| Status below | automatic | recorded by hand |

A test that passes when the hardware is absent proves nothing about the
hardware, so `scripts/verify_device.py` is **not** part of the test suite. It is
a script you run and whose output you record.

## Headless support

Whether a driver produces packets with nothing attached, verified by
`tests/hardware/test_headless_capture.py`:

| Platform | Registered as | Headless mock | Mechanism |
|---|---|---|---|
| Intel 5300 | `intel_5300`, `intel5300` | yes | synthetic packets; file replay via `csiread` |
| ESP32 | `esp32` | yes | synthetic packets, no serial port opened |
| Atheros AR9300 | `atheros_ar9300`, `atheros` | yes | synthetic packets |
| Qualcomm | `qualcomm` | **no** | TCP to an Android device; no mock mode exists |
| Broadcom | — | — | not implemented; see `PLANNED_HARDWARE` |
| MediaTek | — | — | not implemented; see `PLANNED_HARDWARE` |

ESP32 only gained headless support recently: `connect()` previously opened a
serial port regardless of mode, so the mock driver failed on any machine
without a board attached. That left **no** driver able to run unattended, which
is also why a container cannot yet run a streaming workload
(see [deployment_status.md](deployment_status.md)).

**Qualcomm is the remaining gap.** With no mock mode, its capture path cannot
be exercised without an Android device on the network, so it is the least
verifiable platform in the package.

## Running a real-device verification

```bash
python scripts/verify_device.py --hardware esp32 --port /dev/ttyUSB0 --packets 200
```

Nine checks, each reporting PASS, FAIL or SKIP with the observed value:

1. connects to the device
2. reports hardware info
3. calibrates
4. captures the requested packet count
5. packet shape is stable across the capture
6. packets pass `validate_csi_data` (finite, non-negative amplitude, phase in
   range, plausible timestamp)
7. timestamps advance monotonically, with the observed rate against the
   configured one
8. amplitude varies between packets — a frozen capture usually means a stalled
   reader rather than a still room
9. disconnects cleanly

Exit codes: `0` all passed, `1` at least one failed, `2` usage error.

## Per-platform procedure

### Intel 5300

Needs the modified `iwlwifi` driver and the Linux 802.11n CSI Tool. Not usable
on stock kernels.

```bash
sudo modprobe -r iwlwifi && sudo modprobe iwlwifi connector_log=0x1
python scripts/verify_device.py --hardware intel_5300 --packets 500
```

Expect 30 subcarriers, up to 3×3 antennas, rates to ~1 kHz.
Common failure: `connector_log` unset, giving a connection but no packets —
check 4 fails while check 1 passes.

### ESP32

Needs CSI-enabled firmware flashed to the board. Two header formats are
supported (`v1`, `v2`); the driver detects the version on connect in real mode.

```bash
python scripts/verify_device.py --hardware esp32 --port /dev/ttyUSB0 --baud 921600
```

Expect 64 subcarriers (or 128 on some builds), 1–2 antennas, 100–500 Hz.
Common failures: wrong baud rate — check 1 passes, check 4 fails or yields
malformed packets; a serial port held by another process — check 1 fails.

### Atheros AR9300

Needs the Atheros CSI Tool and a patched `ath9k`.

```bash
python scripts/verify_device.py --hardware atheros_ar9300 --packets 500
```

Expect 56 subcarriers, up to 3 antennas.

### Qualcomm

Needs an Android device with CSI extraction running and reachable over TCP.

```bash
python scripts/verify_device.py --hardware qualcomm --device-ip 192.168.1.42
```

Check 1 fails immediately without `--device-ip`, which is correct: there is no
fallback and the driver does not pretend otherwise.

## Verification record

Fill this in as devices are tested. Empty means unverified — not working.

| Platform | Device / firmware | Host OS | Date | Result | Notes |
|---|---|---|---|---|---|
| Intel 5300 | | | | not yet run | |
| ESP32 | | | | not yet run | |
| Atheros AR9300 | | | | not yet run | |
| Qualcomm | | | | not yet run | |

No platform has a recorded real-device run. Everything currently claimed rests
on mock coverage, which is exactly the distinction this page exists to make
visible.

## Known unsupported combinations

- **Intel 5300 on stock kernels** — requires the patched `iwlwifi`; recent
  kernels have dropped support for the original patch series.
- **ESP32 without CSI firmware** — the stock Arduino/ESP-IDF build does not
  emit CSI records; the port opens and no packets arrive.
- **Any platform on Windows for real capture** — the Intel 5300 and Atheros
  paths need Linux kernel modules. ESP32 works over serial on Windows.
- **macOS** — no supported capture path.
- **Containers** — `--device` passthrough can expose a serial device to the
  ESP32 driver, but the kernel-module platforms cannot be containerised.
