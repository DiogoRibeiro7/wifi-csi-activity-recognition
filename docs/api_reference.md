# API Reference

The reference documentation is built with **Sphinx** and the
`autodoc` extension. Each section below mirrors a top‑level package module and
is generated from in‑line docstrings.

## Hardware

```{eval-rst}
.. automodule:: wifi_activity_recognition.hardware
   :members:
   :undoc-members:
```

### Example

```python
from wifi_activity_recognition.hardware import Intel5300Reader
reader = Intel5300Reader(...)
reader.connect()
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

## Utilities

```{eval-rst}
.. automodule:: wifi_activity_recognition.utils
   :members:
```

## Deployment Helpers

```{eval-rst}
.. automodule:: wifi_activity_recognition.inference
   :members:
```

These directives produce a complete API when building the docs with
`sphinx-build docs docs/_build`.
