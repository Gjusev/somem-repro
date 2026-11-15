"""SoMem repro: self-organized document memory (arXiv:2611.11929).

Independent, unofficial, stdlib-only reimplementation of the core mechanism.
Not affiliated with the paper authors (official code: github.com/Qi202/SoMem).
"""

from .organizer import (
    EditError,
    MemoryWorkspace,
    Organizer,
    SourceInteraction,
    TopicDoc,
)
from .views import relationship_graph, recent_view, timeline_view
from .retrieval import EvidenceReader, search
from .agent import (
    BillingError,
    StubSummarizer,
    StubReader,
    ZaiConfig,
    ZaiLLM,
    answer_question,
    build_memory,
)

__version__ = "0.1.0"

__all__ = [
    "BillingError",
    "EditError",
    "EvidenceReader",
    "MemoryWorkspace",
    "Organizer",
    "SourceInteraction",
    "StubReader",
    "StubSummarizer",
    "TopicDoc",
    "ZaiConfig",
    "ZaiLLM",
    "answer_question",
    "build_memory",
    "recent_view",
    "relationship_graph",
    "search",
    "timeline_view",
]
