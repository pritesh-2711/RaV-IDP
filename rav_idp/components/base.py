"""Shared base classes and protocols for pipeline components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class Component(ABC, Generic[TIn, TOut]):
    """Generic base class for composable pipeline components."""

    @abstractmethod
    def run(self, payload: TIn, **kwargs) -> TOut:
        """Execute the component."""


class RegionProcessor(ABC):
    """Base class for processors that operate on detected regions."""

    @abstractmethod
    def supports(self, entity_type: str) -> bool:
        """Return whether the processor supports the supplied entity type."""
