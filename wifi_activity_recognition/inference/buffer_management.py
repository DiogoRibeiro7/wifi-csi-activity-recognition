"""Lock-free circular buffers for streaming CSI data."""

from __future__ import annotations

from collections import deque
from typing import Generic, Iterable, Optional, TypeVar

T = TypeVar("T")


class CircularBuffer(Generic[T]):
    """A simple lock-free circular buffer."""

    def __init__(self, capacity: int) -> None:
        """Initialize the buffer with a maximum capacity."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buffer: deque[T] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """Maximum number of elements the buffer can hold."""
        return self._buffer.maxlen or 0

    def append(self, item: T) -> bool:
        """Append an item, returning ``True`` if an old item was dropped."""
        was_full = len(self._buffer) == self.capacity
        self._buffer.append(item)
        return was_full

    def pop(self) -> Optional[T]:
        """Pop the oldest item from the buffer."""
        try:
            return self._buffer.popleft()
        except IndexError:
            return None

    def extend(self, items: Iterable[T]) -> None:
        """Extend the buffer with an iterable of items."""
        for item in items:
            self.append(item)

    def clear(self) -> None:
        """Remove all items from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return the current number of stored elements."""
        return len(self._buffer)

    def is_full(self) -> bool:
        """Check whether the buffer has reached its capacity."""
        return len(self._buffer) == self.capacity
