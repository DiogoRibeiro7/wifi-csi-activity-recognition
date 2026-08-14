"""Entry point for ``python -m wifi_activity_recognition``.

The Docker images invoke the package this way, but no ``__main__`` existed, so
both the ``prod`` and ``edge`` containers exited immediately with::

    No module named wifi_activity_recognition.__main__;
    'wifi_activity_recognition' is a package and cannot be directly executed

Delegating to the Click group gives ``python -m`` the same surface as the
installed console scripts.
"""

from .cli import cli

if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    cli()
