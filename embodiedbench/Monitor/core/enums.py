from enum import Enum


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EpistemicState(Enum):
    NKK = "known_known"
    NKU = "known_unknown"
    UNK = "unknown_unknown"
    OOD = "out_of_distribution"


class GateDecision(Enum):
    ALLOW = "ALLOW"
    ALLOW_WITH_CONSTRAINTS = "ALLOW_WITH_CONSTRAINTS"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    GATHER_MORE_INFORMATION = "GATHER_MORE_INFORMATION"
    REPLAN = "REPLAN"
    SAFE_FALLBACK = "SAFE_FALLBACK"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
    BLOCK = "BLOCK"
