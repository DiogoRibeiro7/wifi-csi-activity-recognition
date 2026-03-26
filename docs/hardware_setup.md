# Hardware Setup

This guide reflects the hardware drivers that are currently registered in the package: Intel 5300, ESP32, Atheros AR9300, and Qualcomm.

## Check the current registry

Before device-specific setup, confirm what the installed environment exposes:

```bash
python -m wifi_activity_recognition.cli info --hardware all
```

## Intel 5300

1. Install the modified driver and firmware from the Linux 5300 CSI Tool project.
2. Verify the interface is visible with `iw dev`.
3. Confirm the package can instantiate the reader:

```bash
python -m wifi_activity_recognition.cli stream \
  --hardware intel_5300 \
  --duration 5
```

## ESP32

1. Flash an ESP32 board with CSI-capable firmware.
2. Configure the serial connection through `additional_params` in your config file.
3. Test collection:

```bash
python -m wifi_activity_recognition.cli collect \
  --hardware esp32 \
  --packets 10 \
  --output esp32_capture.json
```

## Atheros AR9300

1. Ensure the `ath9k` driver is available and the interface is in monitor mode.
2. Test collection with the registered alias:

```bash
python -m wifi_activity_recognition.cli collect \
  --hardware atheros_ar9300 \
  --packets 10 \
  --output atheros_capture.h5
```

The shorter alias `atheros` may also be available, depending on how the registry was loaded.

## Qualcomm

1. Verify the Qualcomm reader is available in your environment:

```bash
python -m wifi_activity_recognition.cli info --hardware qualcomm
```

2. If the driver is registered successfully, use `collect`, `stream`, or `live` with `--hardware qualcomm`.

## Notes

- Broadcom and MediaTek are not currently enabled in the active hardware registry.
- The current CLI does not provide `listen` or `calibrate` commands.
- The current CLI uses `--hardware`, not `--device`.
