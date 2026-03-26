# API Reference

The API documentation is built with Sphinx autodoc and follows the current public package surface.

## Top-level package

```{eval-rst}
.. automodule:: wifi_activity_recognition
   :members:
```

## Hardware

```{eval-rst}
.. automodule:: wifi_activity_recognition.hardware
   :members:
   :undoc-members:
```

### Example

```python
from wifi_activity_recognition.hardware import CSIReader

reader = CSIReader("esp32", {"sampling_rate": 100, "channel": 6})
```

## Preprocessing

```{eval-rst}
.. automodule:: wifi_activity_recognition.preprocessing
   :members:
```

## Features

```{eval-rst}
.. automodule:: wifi_activity_recognition.features
   :members:
```

## Models

```{eval-rst}
.. automodule:: wifi_activity_recognition.models
   :members:
```

## Datasets

```{eval-rst}
.. automodule:: wifi_activity_recognition.datasets
   :members:
```

## Training

```{eval-rst}
.. automodule:: wifi_activity_recognition.training
   :members:
```

## Inference

```{eval-rst}
.. automodule:: wifi_activity_recognition.inference
   :members:
```

## Utilities

```{eval-rst}
.. automodule:: wifi_activity_recognition.utils
   :members:
```

Build the docs with:

```bash
sphinx-build -b html docs docs/_build/html
```
