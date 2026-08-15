"""Version information for wifi-activity-recognition package."""

__version__ = "0.1.0"

# Version components
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0
VERSION_SUFFIX = ""  # e.g., "alpha", "beta", "rc1"

# Build version tuple
VERSION_TUPLE = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)

# Full version string
if VERSION_SUFFIX:
    __version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}-{VERSION_SUFFIX}"
else:
    __version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"

# Development status
DEVELOPMENT_STATUS = "Alpha"  # Alpha, Beta, Stable


def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "version_tuple": VERSION_TUPLE,
        "development_status": DEVELOPMENT_STATUS,
        "major": VERSION_MAJOR,
        "minor": VERSION_MINOR,
        "patch": VERSION_PATCH,
        "suffix": VERSION_SUFFIX,
    }
