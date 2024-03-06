#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Wed February 28 2024
# =============================================================================
"""Module for constants."""
# =============================================================================

from enum import Enum

METRICS_DICT = {
    "NRA": "Non-rejected accuracy",
    "CQ": "Classification quality",
    "RQ": "Rejection quality",
}

ENTROPY_UNC_LIST = ["TU", "AU", "EU"]
ALL_UNC_LIST = ["TU", "AU", "EU", "confidence"]
GENERAL_UNC_LIST = ["entropy", "confidence"]
UNCERTAINTIES_DICT = {
    "TU": "Total uncertainty",
    "AU": "Aleatoric uncertainty",
    "EU": "Epistemic uncertainty",
    "confidence": "Confidence",
}


class EntropyUnc(str, Enum):
    TU = "TU"
    AU = "AU"
    EU = "EU"


class GeneralUnc(str, Enum):
    ENTROPY = "entropy"
    CONFIDENCE = "confidence"


class AllUnc(str, Enum):
    TU = "TU"
    AU = "AU"
    EU = "EU"
    CONFIDENCE = "confidence"


class Metric(str, Enum):
    NRA = "NRA"
    CQ = "CQ"
    RQ = "RQ"
