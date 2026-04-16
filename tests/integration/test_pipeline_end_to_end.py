from __future__ import annotations

from rav_idp.pipeline import RaVIDPPipeline


def test_pipeline_instantiates() -> None:
    pipeline = RaVIDPPipeline()
    assert pipeline.settings.threshold_table > 0
