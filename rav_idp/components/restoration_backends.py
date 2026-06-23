"""Backend boundary for deterministic and later learned restoration methods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..models import RestorationBackendResult, RestorationPlan
from .page_restorer import restore_page


class RestorationBackend(Protocol):
    name: str

    def restore(self, source_page_v1: bytes, plan: RestorationPlan) -> RestorationBackendResult: ...


@dataclass(frozen=True)
class DeterministicRestorationBackend:
    name: str = "deterministic-opencv-v1"

    def restore(self, source_page_v1: bytes, plan: RestorationPlan) -> RestorationBackendResult:
        image_bytes, mapping = restore_page(source_page_v1, plan)
        return RestorationBackendResult(
            image_bytes=image_bytes,
            mapping=mapping,
            backend_name=self.name,
        )


@dataclass(frozen=True)
class CallableRestorationBackend:
    """Generic adapter for experimental restoration functions."""

    name: str
    restore_fn: Callable[[bytes, RestorationPlan], RestorationBackendResult]

    def restore(self, source_page_v1: bytes, plan: RestorationPlan) -> RestorationBackendResult:
        return self.restore_fn(source_page_v1, plan)
