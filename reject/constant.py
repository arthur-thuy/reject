#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Wed February 28 2024
# =============================================================================
"""Module for constants."""
# =============================================================================

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
