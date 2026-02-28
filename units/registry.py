from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple string -> callable registry used to build pluggable components."""

    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str, item: T) -> None:
        norm_key = key.lower()
        if norm_key in self._items:
            raise KeyError(f"{self.name} registry already has key: {key}")
        self._items[norm_key] = item

    def register_fn(self, key: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            self.register(key, item)
            return item

        return decorator

    def get(self, key: str) -> T:
        norm_key = key.lower()
        if norm_key not in self._items:
            available = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"Unknown {self.name} '{key}'. Available: {available}")
        return self._items[norm_key]

    def has(self, key: str) -> bool:
        return key.lower() in self._items

    def keys(self):
        return sorted(self._items.keys())
