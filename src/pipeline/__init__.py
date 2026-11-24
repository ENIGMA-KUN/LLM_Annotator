"""
Hybrid LLM-assisted annotation pipeline.

Modules:
    input_handler       CSV ingest + schema validation
    annotation_engine   Gemini wrapper + 3-run protocol
    dimension_router    Tier A/B/C splitting logic
    confidence_gate     Inter-run agreement routing
    aggregator          Final score assembly + quality metrics
    quality_monitor     Drift detection + workload metrics
    hybrid_orchestrator  End-to-end hybrid routing protocol
    prompt_payloads     Prometheus-style strict prompt construction
    pairwise_elo        FastChat-style pairwise Elo for subjective dims
"""

from src.pipeline.input_handler import InputHandler
from src.pipeline.annotation_engine import AnnotationEngine
from src.pipeline.dimension_router import DimensionRouter
from src.pipeline.confidence_gate import ConfidenceGate
from src.pipeline.aggregator import PipelineAggregator
from src.pipeline.quality_monitor import QualityMonitor
from src.pipeline.hybrid_orchestrator import HybridOrchestrator
from src.pipeline.prompt_payloads import build_strict_prompt, ANTI_BIAS_PREAMBLE
from src.pipeline.pairwise_elo import PairwiseJudge, EloTournament

__all__ = [
    "InputHandler",
    "AnnotationEngine",
    "DimensionRouter",
    "ConfidenceGate",
    "PipelineAggregator",
    "QualityMonitor",
    "HybridOrchestrator",
    "build_strict_prompt",
    "ANTI_BIAS_PREAMBLE",
    "PairwiseJudge",
    "EloTournament",
]
