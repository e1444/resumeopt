"""Shared strict JSON schema contracts for structured LLM outputs.

Centralized here (rather than duplicated per call site) so parser and
rendering code request the exact same response shape from the LLM instead of
relying purely on prompt-text instructions, which can drift or be ignored.
Schemas follow OpenAI's strict structured-output constraints: every object
lists all of its properties under "required" and sets
"additionalProperties": false at every level.
"""

from __future__ import annotations

from typing import Any, Dict

_CANDIDATE_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "raw_term": {"type": "string"},
        "category": {
            "type": "string",
            "enum": [
                "tool",
                "language",
                "framework",
                "method",
                "domain",
                "certification",
                "soft_skill",
                "responsibility",
                "quality",
                "title",
                "generic",
                "unknown",
            ],
        },
        "include_for_resume_skills": {"type": "boolean"},
        "include_for_cache_candidate": {"type": "boolean"},
        "reason": {"type": "string"},
        "evidence_quote": {"type": "string"},
    },
    "required": [
        "raw_term",
        "category",
        "include_for_resume_skills",
        "include_for_cache_candidate",
        "reason",
        "evidence_quote",
    ],
    "additionalProperties": False,
}

EXTRACTION_CANDIDATES_JSON_SCHEMA: Dict[str, Any] = {
    "name": "extraction_candidates",
    "schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": _CANDIDATE_ITEM_SCHEMA,
            },
        },
        "required": ["candidates"],
        "additionalProperties": False,
    },
}

CHUNK_SPLIT_JSON_SCHEMA: Dict[str, Any] = {
    "name": "chunk_split",
    "schema": {
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["chunks"],
        "additionalProperties": False,
    },
}

GROUNDING_JSON_SCHEMA: Dict[str, Any] = {
    "name": "grounding_check",
    "schema": {
        "type": "object",
        "properties": {
            "is_grounded": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["is_grounded", "reason"],
        "additionalProperties": False,
    },
}

SKILL_GROUPING_JSON_SCHEMA: Dict[str, Any] = {
    "name": "skill_grouping",
    "schema": {
        "type": "object",
        "properties": {
            "active_sections": {
                "type": "array",
                "items": {"type": "string", "enum": ["Languages", "ML & Data", "Tools"]},
            },
            "grouped_skills": {
                "type": "object",
                "properties": {
                    "Languages": {"type": "array", "items": {"type": "string"}},
                    "ML & Data": {"type": "array", "items": {"type": "string"}},
                    "Tools": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["Languages", "ML & Data", "Tools"],
                "additionalProperties": False,
            },
        },
        "required": ["active_sections", "grouped_skills"],
        "additionalProperties": False,
    },
}
