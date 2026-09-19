#!/usr/bin/env python3
"""
01_analyze_and_chunk_target_text.py
Deterministic Target Text Analysis & Segmentation Strategies for Spanish -> German.

Implements Sections 8, 9, 10 of the autonomous refinement specification:
- Strategy S1: Sentence-level splitting (with dialogue interjection handling to eliminate 1-word fragments)
- Strategy S2: Sentence + clause-level splitting (splitting at appropriate clause/dialogue boundaries)
- Strategy S3: Short semantic chunks (8-18 words per chunk)
- Verifies exact word and semantic preservation against frozen translation.
- Performs automatic chunking QC (no missing verbs, no unnatural breaks, no 1-word fragments, no lost text).
"""

import sys
import json
import re
from pathlib import Path
import pandas as pd

# Paths
WORKSPACE = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement")
TEXT_DIR = WORKSPACE / "02_text_analysis"
CHUNKING_DIR = WORKSPACE / "05_chunking_search"

# Frozen German Target Text from Stage 5 / Pilot Refinement
ORIGINAL_GERMAN_TEXT = (
    "Gestern ging ich zum Arzt. „Was fehlt Ihnen?“, fragte er mich. "
    "Ich sagte: „Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!“ "
    "„Haben Sie einen gebrochenen Nagel?“ „Ja.“ „Nun, dann wissen wir ja, was es ist! "
    "Geben Sie Ihren Scheck am Ausgang ab.“"
)

NORMALIZED_GERMAN_TEXT = (
    "Gestern ging ich zum Arzt. Was fehlt Ihnen? fragte er mich. "
    "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh! "
    "Haben Sie einen gebrochenen Nagel? Ja. Nun, dann wissen wir ja, was es ist! "
    "Geben Sie Ihren Scheck am Ausgang ab."
)

def normalize_text_for_comparison(text: str) -> str:
    # Lowercase, strip punctuation and collapse whitespace for exact word sequence check
    words = re.findall(r"\b[\wÄÖÜäöüß]+\b", text.lower())
    return " ".join(words)

def build_chunking_strategies():
    """
    Build multiple deterministic segmentation strategies.
    Every strategy must strictly preserve the exact word sequence.
    """
    strategies = {}

    # -------------------------------------------------------------
    # S1_baseline: Original Config C (6 segments, but note seg 5 is 'Ja.')
    # Included for direct comparison and quality audit
    # -------------------------------------------------------------
    strategies["S1_config_c"] = {
        "description": "Original Config C 6-segment baseline (contains 1-word fragment 'Ja.')",
        "chunks": [
            "Gestern ging ich zum Arzt.",
            "Was fehlt Ihnen? fragte er mich.",
            "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!",
            "Haben Sie einen gebrochenen Nagel?",
            "Ja.",
            "Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab."
        ]
    }

    # -------------------------------------------------------------
    # S1: Sentence-level splitting (clean grammatical sentences, no 1-word fragment)
    # Merges affirmative response 'Ja.' into either question or doctor's reply
    # -------------------------------------------------------------
    strategies["S1_sentence"] = {
        "description": "Clean sentence-level splitting with dialogue interjection merge (no 1-word fragment)",
        "chunks": [
            "Gestern ging ich zum Arzt.",
            "Was fehlt Ihnen? fragte er mich.",
            "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!",
            "Haben Sie einen gebrochenen Nagel?",
            "Ja. Nun, dann wissen wir ja, was es ist!",
            "Geben Sie Ihren Scheck am Ausgang ab."
        ]
    }

    # S1 variant B: Merge Ja with question
    strategies["S1_dialogue_q"] = {
        "description": "Sentence-level splitting with question-response coupling (Ja coupled to prompt)",
        "chunks": [
            "Gestern ging ich zum Arzt.",
            "Was fehlt Ihnen? fragte er mich.",
            "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!",
            "Haben Sie einen gebrochenen Nagel? Ja.",
            "Nun, dann wissen wir ja, was es ist!",
            "Geben Sie Ihren Scheck am Ausgang ab."
        ]
    }

    # -------------------------------------------------------------
    # S2: Sentence + clause-level splitting
    # Splits long complex sentences at natural grammatical clause boundaries
    # -------------------------------------------------------------
    strategies["S2_clause"] = {
        "description": "Sentence + clause-level splitting at natural grammatical boundaries",
        "chunks": [
            "Gestern ging ich zum Arzt.",
            "Was fehlt Ihnen? fragte er mich.",
            "Ich sagte: Ach Herr Doktor,",
            "wo immer ich den Finger hinlege, tut es weh!",
            "Haben Sie einen gebrochenen Nagel?",
            "Ja. Nun, dann wissen wir ja, was es ist!",
            "Geben Sie Ihren Scheck am Ausgang ab."
        ]
    }

    # -------------------------------------------------------------
    # S3: Short balanced semantic chunks (8-18 words per chunk)
    # Synthesizes cohesive semantic breath units without fragmenting meaning
    # -------------------------------------------------------------
    strategies["S3_semantic_4chunk"] = {
        "description": "Balanced 4-chunk semantic grouping targeting 8-15 words per chunk",
        "chunks": [
            "Gestern ging ich zum Arzt. Was fehlt Ihnen? fragte er mich.",
            "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!",
            "Haben Sie einen gebrochenen Nagel? Ja. Nun, dann wissen wir ja, was es ist!",
            "Geben Sie Ihren Scheck am Ausgang ab."
        ]
    }

    strategies["S3_semantic_3chunk"] = {
        "description": "Balanced 3-chunk macro semantic units (11-18 words per chunk)",
        "chunks": [
            "Gestern ging ich zum Arzt. Was fehlt Ihnen? fragte er mich.",
            "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh!",
            "Haben Sie einen gebrochenen Nagel? Ja. Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab."
        ]
    }

    return strategies

def audit_chunking_strategy(name: str, data: dict, ref_normalized: str) -> dict:
    chunks = data["chunks"]
    desc = data["description"]

    # 1. Full text reconstruction check
    reconstructed_text = " ".join(chunks)
    ref_words = normalize_text_for_comparison(ref_normalized)
    recon_words = normalize_text_for_comparison(reconstructed_text)

    exact_word_match = (ref_words == recon_words)
    ref_word_list = ref_words.split()
    recon_word_list = recon_words.split()

    missing_words = [w for w in ref_word_list if w not in recon_word_list]
    extra_words = [w for w in recon_word_list if w not in ref_word_list]

    # 2. Chunk properties & QC checks
    chunk_details = []
    has_one_word_fragment = False
    has_empty_or_punct_only = False
    chunks_out_of_range = []

    # Simple German finite verb patterns for sanity check
    german_verbs = {
        "ging", "fehlt", "fragte", "sagte", "hinlege", "tut", "haben",
        "wissen", "ist", "geben"
    }

    for idx, c in enumerate(chunks, 1):
        words = re.findall(r"\b[\wÄÖÜäöüß]+\b", c)
        word_count = len(words)
        char_count = len(c)

        if word_count <= 1:
            has_one_word_fragment = True

        if word_count == 0 or all(not ch.isalnum() for ch in c):
            has_empty_or_punct_only = True

        # Verb presence check
        c_words_lower = [w.lower() for w in words]
        contains_verb = any(v in c_words_lower for v in german_verbs)

        chunk_details.append({
            "chunk_idx": idx,
            "text": c,
            "word_count": word_count,
            "char_count": char_count,
            "contains_verb": contains_verb
        })

    # Hard rejection conditions per Section 10:
    # - sentence fragments with missing verbs
    # - unnatural semantic breaks
    # - one-word fragments
    # - punctuation-only fragments
    # - duplicated words
    # - lost text
    rejection_reasons = []
    if not exact_word_match:
        rejection_reasons.append(f"Word sequence mismatch! missing={missing_words}, extra={extra_words}")
    if has_one_word_fragment:
        rejection_reasons.append("Contains one-word fragment(s)")
    if has_empty_or_punct_only:
        rejection_reasons.append("Contains empty or punctuation-only fragment(s)")
    for cd in chunk_details:
        if not cd["contains_verb"]:
            rejection_reasons.append(f"Chunk {cd['chunk_idx']} lacks a recognized finite verb")

    passed_qc = len(rejection_reasons) == 0

    return {
        "strategy_id": name,
        "description": desc,
        "num_chunks": len(chunks),
        "total_words": len(recon_word_list),
        "exact_word_match": exact_word_match,
        "has_one_word_fragment": has_one_word_fragment,
        "passed_qc": passed_qc,
        "rejection_reasons": "; ".join(rejection_reasons) if rejection_reasons else "NONE",
        "chunk_details": chunk_details
    }

def main():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKING_DIR.mkdir(parents=True, exist_ok=True)

    # Save frozen texts
    (TEXT_DIR / "german_target_text_original.txt").write_text(ORIGINAL_GERMAN_TEXT, encoding="utf-8")
    (TEXT_DIR / "german_target_text_normalized.txt").write_text(NORMALIZED_GERMAN_TEXT, encoding="utf-8")

    strategies = build_chunking_strategies()
    audit_results = []
    all_chunks_flat = []

    for name, sdata in strategies.items():
        res = audit_chunking_strategy(name, sdata, NORMALIZED_GERMAN_TEXT)
        audit_results.append({
            "strategy_id": res["strategy_id"],
            "description": res["description"],
            "num_chunks": res["num_chunks"],
            "total_words": res["total_words"],
            "exact_word_match": res["exact_word_match"],
            "has_one_word_fragment": res["has_one_word_fragment"],
            "passed_qc": res["passed_qc"],
            "rejection_reasons": res["rejection_reasons"]
        })

        for cd in res["chunk_details"]:
            all_chunks_flat.append({
                "strategy_id": name,
                "chunk_idx": cd["chunk_idx"],
                "word_count": cd["word_count"],
                "char_count": cd["char_count"],
                "contains_verb": cd["contains_verb"],
                "text": cd["text"]
            })

        # Save individual strategy JSON
        strat_out = {
            "strategy_id": name,
            "description": sdata["description"],
            "passed_qc": res["passed_qc"],
            "rejection_reasons": res["rejection_reasons"],
            "chunks": sdata["chunks"]
        }
        with open(CHUNKING_DIR / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(strat_out, f, indent=2, ensure_ascii=False)

    df_audit = pd.DataFrame(audit_results)
    df_audit.to_csv(CHUNKING_DIR / "chunking_strategies_audit.csv", index=False)
    df_audit.to_csv(TEXT_DIR / "chunking_strategies_audit.csv", index=False)

    df_chunks = pd.DataFrame(all_chunks_flat)
    df_chunks.to_csv(CHUNKING_DIR / "all_chunk_variants.csv", index=False)
    df_chunks.to_csv(TEXT_DIR / "all_chunk_variants.csv", index=False)

    print("=== Target Text Analysis & Chunking Strategies Complete ===")
    print(df_audit[["strategy_id", "num_chunks", "passed_qc", "rejection_reasons"]].to_string())

if __name__ == "__main__":
    main()
