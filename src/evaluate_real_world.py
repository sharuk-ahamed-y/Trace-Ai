"""
src/evaluate_real_world.py — Controlled Real-World Benchmark Evaluation Runner

Safeguards:
1. Keeps real-world evaluation strictly separate from synthetic held-out metrics (SIPGD_v2).
2. If no authorized real-world dataset/manifest exists, exits safely and reports PREPARED BUT NOT EXECUTED.
3. Does NOT fabricate, hallucinate, or invent performance numbers.
4. Validates source-grouped partition independence and independent ground truth.
5. Rejects malformed, duplicate, inconsistent, or unauthorized inputs with actionable error messages.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
from PIL import Image

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VALID_DIRECTIONS = {"A_TO_B", "B_TO_A", "UNCERTAIN", "NONE"}
SUPPORTED_TRANSFORMATIONS = [
    "crop", "resize", "jpeg", "blur", "brightness", "contrast",
    "sharpen", "rotation", "color", "noise", "perspective", "quality_degradation"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TRACE AI models on authorized real-world image benchmark.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "real_world_benchmark" / "real_pairs_manifest.csv"),
        help="Path to authorized ground-truth CSV manifest."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "real_world_benchmark" / "images"),
        help="Directory containing benchmark images."
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "REAL_WORLD_EVALUATION.md"),
        help="Output report markdown path."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify readiness, manifest schema, and partition integrity without executing inference."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize actual model evaluation on the benchmark. Default is safe readiness mode."
    )
    return parser.parse_args()


def compute_ece(probs: List[float], labels: List[int], n_bins: int = 10) -> float:
    """Computes Expected Calibration Error across n_bins for binary classification."""
    import numpy as np
    probs = np.array(probs, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def parse_transform_list(raw_val: Any) -> Set[str]:
    """Parses transformation string/list into a normalized set of strings."""
    if raw_val is None or (isinstance(raw_val, float) and str(raw_val) == "nan"):
        return set()
    s = str(raw_val).strip()
    if not s or s.lower() in ("none", "[]", "{}"):
        return set()
    if s.startswith("[") and s.endswith("]"):
        try:
            items = json.loads(s)
            return {str(x).strip().lower() for x in items if str(x).strip().lower() not in ("none", "")}
        except Exception:
            pass
    parts = s.replace(";", ",").split(",")
    return {p.strip().lower() for p in parts if p.strip().lower() not in ("none", "")}


def classify_pair_category(row) -> str:
    """Classifies a benchmark pair into one of three isolated categories."""
    pid = str(row.get("pair_id", "")).strip()
    notes = str(row.get("notes", "")).strip().lower()
    rel = int(row.get("related", 0))
    if pid == "pair_0001" or "contributor-confirmed" in notes:
        return "contributor_confirmed"
    elif pid.startswith("ct_pair_") or "controlled_transform" in notes:
        return "controlled_transform"
    elif pid.startswith("neg_pair_") or "cross_source_negative" in notes or rel == 0:
        return "cross_scene_negative"
    else:
        return "unclassified"


def validate_manifest(df, images_dir: Path, check_images: bool = False) -> Tuple[bool, List[str]]:
    """
    Validates manifest schema, duplicate entries, data types, label consistency,
    file extensions, path traversal, category classification, and source-grouped split independence.
    If check_images is True, validates physical existence and PIL readability of every image.
    Returns (is_valid, list_of_error_messages).
    """
    errors = []

    # 1. Required Columns
    required_cols = {"pair_id", "image_a", "image_b", "source_a", "source_b", "related", "ground_truth_direction"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(list(missing_cols))}")
        return False, errors

    # 2. Duplicate Pair IDs
    dup_ids = df[df["pair_id"].duplicated()]["pair_id"].tolist()
    if dup_ids:
        errors.append(f"Duplicate pair_id entries found: {dup_ids[:5]} (total {len(dup_ids)})")

    # 3. Duplicate Image Pairs (Order-agnostic check)
    seen_pairs = set()
    for idx, row in df.iterrows():
        a = str(row["image_a"]).strip()
        b = str(row["image_b"]).strip()
        pair_key = tuple(sorted([a, b]))
        if pair_key in seen_pairs:
            errors.append(f"Duplicate pair detected at row {idx}: {a} and {b}")
        seen_pairs.add(pair_key)

    # 4. Value Range & Semantic Consistency
    resolved_dir = images_dir.resolve()
    for idx, row in df.iterrows():
        pid = row["pair_id"]
        rel = row["related"]
        # Related must be 0 or 1
        if rel not in (0, 1, "0", "1"):
            errors.append(f"Row {idx} ({pid}): 'related' value must be 0 or 1, got '{rel}'")
            continue
        rel_int = int(rel)

        src_a = str(row["source_a"]).strip()
        src_b = str(row["source_b"]).strip()
        if not src_a or not src_b:
            errors.append(f"Row {idx} ({pid}): source_a and source_b must be non-empty strings")

        # Consistency: related == 1 <=> source_a == source_b
        if rel_int == 1 and src_a != src_b:
            errors.append(f"Row {idx} ({pid}): 'related' is 1 but source_a ('{src_a}') != source_b ('{src_b}')")
        if rel_int == 0 and src_a == src_b:
            errors.append(f"Row {idx} ({pid}): 'related' is 0 but source_a == source_b ('{src_a}')")

        # Direction check
        dir_val = str(row["ground_truth_direction"]).strip().upper()
        if dir_val not in VALID_DIRECTIONS:
            errors.append(f"Row {idx} ({pid}): invalid ground_truth_direction '{dir_val}' (must be one of {VALID_DIRECTIONS})")

        if rel_int == 0 and dir_val != "NONE":
            errors.append(f"Row {idx} ({pid}): 'related' is 0 but direction is '{dir_val}' (must be 'NONE')")

        if rel_int == 1 and dir_val == "NONE":
            errors.append(f"Row {idx} ({pid}): 'related' is 1 but direction is 'NONE' (must be A_TO_B, B_TO_A, or UNCERTAIN)")

        # Category classification check (reject unclassified rows unless explicitly defined)
        cat = classify_pair_category(row)
        if cat == "unclassified":
            notes_str = str(row.get("notes", "")).strip().lower()
            if "placeholder" in str(pid).lower() or "placeholder" in notes_str or "template" in notes_str:
                pass  # Explicitly defined template/placeholder row
            else:
                errors.append(f"Row {idx} ({pid}): unclassified pair row (does not match contributor_confirmed, controlled_transform, or cross_scene_negative)")

        # File path, path traversal, extension & readability checks
        for col_name in ["image_a", "image_b"]:
            fname = str(row[col_name]).strip()
            if ".." in fname or fname.startswith("/") or fname.startswith("\\"):
                errors.append(f"Row {idx} ({pid}): invalid relative path in {col_name}: '{fname}'")

            resolved_img = (images_dir / fname).resolve()
            try:
                resolved_img.relative_to(resolved_dir)
            except ValueError:
                errors.append(f"Row {idx} ({pid}): path traversal detected in {col_name}: '{fname}'")

            ext = Path(fname).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                errors.append(f"Row {idx} ({pid}): unsupported file extension in {col_name}: '{fname}' ({ext})")

            if check_images:
                if not resolved_img.exists():
                    errors.append(f"Row {idx} ({pid}): image file missing: {col_name}='{fname}'")
                elif not resolved_img.is_file():
                    errors.append(f"Row {idx} ({pid}): image path is not a file: {col_name}='{fname}'")
                else:
                    try:
                        with Image.open(resolved_img) as img:
                            img.verify()
                    except Exception as img_err:
                        errors.append(f"Row {idx} ({pid}): image file unreadable/corrupted in {col_name} '{fname}': {img_err}")

    # 5. Source Group Split Disjointness
    if "split" in df.columns:
        splits = [s for s in df["split"].unique() if str(s).strip() != ""]
        split_sources = {}
        for s in splits:
            s_df = df[df["split"] == s]
            s_sources = set(s_df["source_a"]).union(set(s_df["source_b"]))
            split_sources[s] = s_sources

        for i, s1 in enumerate(splits):
            for s2 in splits[i+1:]:
                overlap = split_sources[s1].intersection(split_sources[s2])
                if overlap:
                    errors.append(
                        f"Source leakage across evaluation splits: sources {sorted(list(overlap))[:5]} "
                        f"appear in both '{s1}' and '{s2}'"
                    )

    return len(errors) == 0, errors


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    images_dir = Path(args.images_dir)
    output_report = Path(args.output_report)

    print("=" * 80)
    print("TRACE AI — Controlled Real-World Benchmark Evaluation Runner")
    print("=" * 80)

    # Safeguard 1: Verify existence of authorized benchmark
    if not manifest_path.exists():
        print(f"[RealWorldEval] No authorized real-world benchmark manifest found at:")
        print(f"                {manifest_path}")
        print("\n[RealWorldEval] Benchmark Status: PREPARED BUT NOT EXECUTED")
        print("[RealWorldEval] Safeguard Enforced: Zero results fabricated or assumed.")
        print("[RealWorldEval] Synthetic held-out metrics (SIPGD_v2) remain strictly isolated.")
        print("=" * 80)
        return 0

    import pandas as pd
    import numpy as np

    try:
        df = pd.read_csv(manifest_path)
    except Exception as ex:
        print(f"[RealWorldEval] ERROR: Failed to read manifest {manifest_path}: {ex}")
        return 1

    if len(df) == 0:
        print(f"[RealWorldEval] Benchmark manifest {manifest_path} is empty (0 pairs).")
        print("[RealWorldEval] Benchmark Status: PREPARED BUT NOT EXECUTED")
        return 0

    # Safeguard 2: Comprehensive Manifest Validation (includes image presence & readability checks)
    is_valid, validation_errors = validate_manifest(df, images_dir, check_images=True)
    if not is_valid:
        print(f"[RealWorldEval] ERROR: Manifest validation failed with {len(validation_errors)} error(s):")
        for err in validation_errors[:10]:
            print(f"  - {err}")
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more errors.")
        return 1

    print(f"[RealWorldEval] Manifest validation: PASSED ({len(df)} pairs, schema, readability, and source-grouping verified).")

    # Subgroup breakdown
    cat_counts = {}
    for _, row in df.iterrows():
        c = classify_pair_category(row)
        cat_counts[c] = cat_counts.get(c, 0) + 1

    print("[RealWorldEval] Benchmark Subgroup Composition:")
    print(f"  - Contributor-Confirmed Real Pairs: {cat_counts.get('contributor_confirmed', 0)}")
    print(f"  - Controlled-Transform Pairs:       {cat_counts.get('controlled_transform', 0)}")
    print(f"  - Cross-Scene Negative Pairs:       {cat_counts.get('cross_scene_negative', 0)}")
    if cat_counts.get("unclassified", 0) > 0:
        print(f"  - Unclassified Pairs:               {cat_counts.get('unclassified', 0)}")

    if args.dry_run:
        print("[RealWorldEval] Dry-run complete. All schema, subgroup isolation, and partition checks passed.")
        return 0

    if not args.execute:
        print("\n[RealWorldEval] Benchmark Status: PREPARED BUT NOT EXECUTED")
        print("[RealWorldEval] Safeguard Enforced: Zero results fabricated or assumed.")
        print("[RealWorldEval] Subgroup Reporting: Enabled (separate sections for contributor, controlled, and negative).")
        print("[RealWorldEval] Notice: Actual evaluation execution requires explicit '--execute' flag and reviewed dataset.")
        return 0

    # Safeguard 3: Verify image directory existence
    if not images_dir.exists():
        print(f"[RealWorldEval] ERROR: Images directory not found: {images_dir}")
        return 1

    # Execute evaluation via ModelManager
    from backend.app.services.model_manager import ModelManager
    manager = ModelManager()
    manager.set_phase4_enabled(True)

    if not manager.phase4_loaded:
        print(f"[RealWorldEval] ERROR: Failed to load Phase 4 models: {manager.phase4_fallback_reason}")
        print("[RealWorldEval] Refusing to run evaluation: heuristic or legacy fallbacks cannot be reported as trained-model performance.")
        return 1

    evaluated_records = []
    expected_count = len(df)
    print(f"[RealWorldEval] Running pairwise inference across {expected_count} pairs...")
    missing_files = 0
    unreadable_files = 0
    threshold = manager.phase4_rel_threshold

    for idx, row in df.iterrows():
        path_a = images_dir / str(row["image_a"]).strip()
        path_b = images_dir / str(row["image_b"]).strip()

        if not path_a.exists() or not path_b.exists():
            print(f"[RealWorldEval] ERROR: Missing image file for pair {row['pair_id']}: '{path_a}' or '{path_b}'")
            missing_files += 1
            break

        try:
            res = manager.predict_pair(path_a, path_b, strict_phase4=True)
        except Exception as ex_inf:
            print(f"[RealWorldEval] ERROR during Phase 4 inference for pair {row['pair_id']}: {ex_inf}")
            unreadable_files += 1
            break

        # Hard safeguard: Ensure genuine Phase 4 trained model was used (no silent heuristic or legacy fallbacks)
        inf_path = res.get("inference_path", {})
        inf_mode = res.get("inference_mode")
        act_model = res.get("active_model", "")
        if (
            inf_path.get("pipeline") != "phase4_v1"
            or inf_path.get("is_fallback") is True
            or inf_mode != "trained_model"
            or "Phase 4" not in act_model
        ):
            print(f"[RealWorldEval] CRITICAL ERROR: Pair {row['pair_id']} evaluated with invalid inference provenance.")
            print(f"                inference_path={inf_path}, active_model='{act_model}', inference_mode='{inf_mode}'")
            return 1

        cat = classify_pair_category(row)
        rel_gt = int(row["related"])
        rel_prob = float(res["related_confidence"])
        pred_rel = int(rel_prob >= threshold)

        dir_gt = str(row["ground_truth_direction"]).strip().upper()
        dir_pred = res["direction"]
        dir_conf = float(res["direction_confidence"])

        gt_tfms = parse_transform_list(row.get("applied_transformations", ""))
        pred_tfms = set(res.get("compound_tags", []))
        if res.get("transformation") and res["transformation"] != "none":
            pred_tfms.add(res["transformation"])

        evaluated_records.append({
            "pair_id": str(row["pair_id"]).strip(),
            "category": cat,
            "image_a": str(row["image_a"]).strip(),
            "image_b": str(row["image_b"]).strip(),
            "source_a": str(row["source_a"]).strip(),
            "source_b": str(row["source_b"]).strip(),
            "gt_rel": rel_gt,
            "pred_rel_prob": rel_prob,
            "pred_rel": pred_rel,
            "gt_dir": dir_gt,
            "pred_dir": dir_pred,
            "pred_dir_conf": dir_conf,
            "gt_tfms": gt_tfms,
            "pred_tfms": pred_tfms,
            "pred_primary_tfm": res.get("transformation", "none"),
            "notes": str(row.get("notes", "")).strip(),
            "inference_path": inf_path
        })

    n_eval = len(evaluated_records)
    failed_count = missing_files + unreadable_files
    if failed_count > 0 or n_eval != expected_count:
        print(f"[RealWorldEval] CRITICAL ERROR: Partial evaluation detected ({n_eval}/{expected_count} pairs evaluated, {failed_count} failed).")
        print("[RealWorldEval] Refusing to generate a complete benchmark report on a partial dataset.")
        return 1

    # Helper functions for metrics
    def calc_binary(recs):
        if not recs:
            return {"total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0, "prec": 0.0, "rec": 0.0, "f1": 0.0, "fpr": 0.0, "acc": 0.0}
        tp = sum(1 for r in recs if r["pred_rel"] == 1 and r["gt_rel"] == 1)
        fp = sum(1 for r in recs if r["pred_rel"] == 1 and r["gt_rel"] == 0)
        tn = sum(1 for r in recs if r["pred_rel"] == 0 and r["gt_rel"] == 0)
        fn = sum(1 for r in recs if r["pred_rel"] == 0 and r["gt_rel"] == 1)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        acc = (tp + tn) / len(recs) if len(recs) > 0 else 0.0
        return {"total": len(recs), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "prec": p, "rec": r, "f1": f, "fpr": fpr, "acc": acc}

    def calc_dir(recs):
        rel_recs = [r for r in recs if r["gt_rel"] == 1]
        if not rel_recs:
            return {"total": 0, "raw_acc": 0.0, "conf_acc": 0.0, "coverage": 0.0, "abstentions": 0}
        corr_raw = sum(1 for r in rel_recs if r["gt_dir"] == r["pred_dir"])
        raw_acc = corr_raw / len(rel_recs)
        conf_recs = [r for r in rel_recs if r["pred_dir"] != "UNCERTAIN"]
        cov = len(conf_recs) / len(rel_recs)
        conf_acc = (sum(1 for r in conf_recs if r["gt_dir"] == r["pred_dir"]) / len(conf_recs)) if conf_recs else 0.0
        return {"total": len(rel_recs), "raw_acc": raw_acc, "conf_acc": conf_acc, "coverage": cov, "abstentions": len(rel_recs) - len(conf_recs)}

    # Subgroups
    cont_records = [r for r in evaluated_records if r["category"] == "contributor_confirmed"]
    ctrl_records = [r for r in evaluated_records if r["category"] == "controlled_transform"]
    neg_records = [r for r in evaluated_records if r["category"] == "cross_scene_negative"]

    cont_m = calc_binary(cont_records)
    cont_dir = calc_dir(cont_records)

    ctrl_m = calc_binary(ctrl_records)
    ctrl_dir = calc_dir(ctrl_records)

    neg_m = calc_binary(neg_records)

    all_m = calc_binary(evaluated_records)
    all_dir = calc_dir(evaluated_records)

    all_rel_true = [r["gt_rel"] for r in evaluated_records]
    all_rel_prob = [r["pred_rel_prob"] for r in evaluated_records]
    brier = float(np.mean((np.array(all_rel_prob) - np.array(all_rel_true)) ** 2))
    ece = compute_ece(all_rel_prob, all_rel_true, n_bins=10)

    # ROC-AUC if both classes present
    roc_auc_val = None
    if len(set(all_rel_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc_val = float(roc_auc_score(all_rel_true, all_rel_prob))
        except Exception:
            pass

    # Multi-label transformation metrics (over positive pairs)
    pos_records = [r for r in evaluated_records if r["gt_rel"] == 1]
    tfm_stats = {}
    macro_f1 = None
    if pos_records:
        per_class_f1 = []
        for c in SUPPORTED_TRANSFORMATIONS:
            tp_c = sum(1 for r in pos_records if c in r["gt_tfms"] and c in r["pred_tfms"])
            fp_c = sum(1 for r in pos_records if c not in r["gt_tfms"] and c in r["pred_tfms"])
            fn_c = sum(1 for r in pos_records if c in r["gt_tfms"] and c not in r["pred_tfms"])
            p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            support_c = sum(1 for r in pos_records if c in r["gt_tfms"])
            tfm_stats[c] = {"precision": round(p_c, 4), "recall": round(r_c, 4), "f1": round(f1_c, 4), "support": support_c}
            if support_c > 0:
                per_class_f1.append(f1_c)
        macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    # Write separate Real-World Report with disaggregated categories
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("# REAL-WORLD BENCHMARK EVALUATION REPORT\n\n")
        f.write("**TRACE AI: Image Provenance Using Graphs**  \n")
        f.write("**Status:** EXECUTED ON AUTHORIZED REAL-WORLD DATASET  \n")
        f.write(f"**Expected Pairs:** {expected_count}  \n")
        f.write(f"**Evaluated Pairs:** {n_eval}  \n")
        f.write(f"**Skipped Pairs:** 0  \n")
        f.write(f"**Failed Pairs:** 0  \n")
        f.write(f"**Active Model:** Phase 4 Normalized Forensic MLP Pipeline (v1) (`inference_path.pipeline: phase4_v1`, `is_fallback: False`)  \n")
        f.write(f"**Decision Threshold ($\\\\tau$):** {threshold} (Validation-tuned)  \n")
        f.write(f"**Direction Uncertainty Delta ($\\\\Delta$):** {manager.phase4_dir_delta_uncertain}  \n\n")

        # Section 1: Contributor-Confirmed Real Pairs
        f.write("## 1. Category A: Contributor-Confirmed Real-World Positive Pairs\n\n")
        f.write(f"Evaluates genuine unconstrained photograph captures with contributor-confirmed physical lineage.\n\n")
        f.write(f"- **Total Pairs:** {len(cont_records)}\n")
        if cont_records:
            f.write(f"- **Relationship Detection:** TP={cont_m['tp']}, FN={cont_m['fn']} (Recall: {cont_m['rec']:.4f})\n")
            f.write(f"- **Direction Accuracy (Raw):** {cont_dir['raw_acc']:.4f}\n")
            f.write(f"- **Direction Coverage:** {cont_dir['coverage']:.4f} ({cont_dir['coverage']*100:.1f}%)\n")
            f.write(f"- **Direction Accuracy (Confident):** {cont_dir['conf_acc']:.4f}\n\n")
            f.write("| Pair ID | Image A | Image B | P(Rel) | Rel Pred | Dir (GT -> Pred) | Conf | Match? | Tfm (GT -> Pred) |\n")
            f.write("|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|\n")
            for r in cont_records:
                dir_match = "PASS" if r["gt_dir"] == r["pred_dir"] else "FAIL"
                tfm_match = "PASS" if r["gt_tfms"].issubset(r["pred_tfms"]) else "PARTIAL/FAIL"
                f.write(f"| `{r['pair_id']}` | `{r['image_a'][:22]}...` | `{r['image_b'][:22]}...` | {r['pred_rel_prob']:.4f} | {r['pred_rel']} | `{r['gt_dir']}` -> `{r['pred_dir']}` | {r['pred_dir_conf']:.4f} | {dir_match} | `{list(r['gt_tfms'])}` -> `{list(r['pred_tfms'])}` |\n")
            f.write("\n")
        else:
            f.write("*No contributor-confirmed pairs evaluated.*\n\n")

        # Section 2: Controlled-Transformation Positive Pairs
        f.write("## 2. Category B: Controlled-Transformation Positive Pairs\n\n")
        f.write(f"Evaluates controlled canonical transformations applied to authorized base photograph (`real_src_0001`).\n\n")
        f.write(f"- **Total Pairs:** {len(ctrl_records)}\n")
        f.write(f"- **Base Physical Image:** `WhatsApp Image 2026-10-13 at 1.31.12 AM.jpeg` (`real_src_0001`)\n")
        if ctrl_records:
            f.write(f"- **Relationship Detection:** TP={ctrl_m['tp']}, FN={ctrl_m['fn']} (Recall: {ctrl_m['rec']:.4f})\n")
            f.write(f"- **Direction Accuracy (Raw):** {ctrl_dir['raw_acc']:.4f}\n")
            f.write(f"- **Direction Coverage:** {ctrl_dir['coverage']:.4f} ({ctrl_dir['coverage']*100:.1f}%)\n")
            f.write(f"- **Direction Accuracy (Confident):** {ctrl_dir['conf_acc']:.4f}\n\n")
            f.write("| Pair ID | Transformation | P(Rel) | Rel Pred | Dir (GT -> Pred) | Conf | Dir Match? | Primary Tfm Pred |\n")
            f.write("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|\n")
            for r in ctrl_records:
                dir_match = "PASS" if r["gt_dir"] == r["pred_dir"] else "FAIL"
                f.write(f"| `{r['pair_id']}` | `{list(r['gt_tfms'])[0] if r['gt_tfms'] else 'none'}` | {r['pred_rel_prob']:.4f} | {r['pred_rel']} | `{r['gt_dir']}` -> `{r['pred_dir']}` | {r['pred_dir_conf']:.4f} | {dir_match} | `{r['pred_primary_tfm']}` |\n")
            f.write("\n")
        else:
            f.write("*No controlled-transformation pairs evaluated.*\n\n")

        # Section 3: Cross-Scene Negative Distractor Pairs
        f.write("## 3. Category C: Cross-Scene Negative Distractor Pairs\n\n")
        f.write(f"Evaluates unrelated image pairs across contributor-confirmed distinct physical scenes (`related=0`).\n\n")
        f.write(f"- **Total Negative Pairs:** {len(neg_records)}\n")
        if neg_records:
            distinct_sources = set(r["source_a"] for r in neg_records).union(set(r["source_b"] for r in neg_records))
            f.write(f"- **Confirmed Physical Scenes Involved:** {len(distinct_sources)}\n")
            f.write(f"- **True Negatives (TN):** {neg_m['tn']} / {len(neg_records)}\n")
            f.write(f"- **False Positives (FP / False Alarms):** {neg_m['fp']} / {len(neg_records)}\n")
            f.write(f"- **Specificity (True Negative Rate):** {neg_m['tn']/len(neg_records):.4f}\n")
            f.write(f"- **False Positive Rate (FPR):** {neg_m['fpr']:.4f}\n")
            max_neg_prob = max(r["pred_rel_prob"] for r in neg_records)
            mean_neg_prob = float(np.mean([r["pred_rel_prob"] for r in neg_records]))
            f.write(f"- **Negative Score Distribution:** Max P(Rel)={max_neg_prob:.4f}, Mean P(Rel)={mean_neg_prob:.4f}\n\n")
        else:
            f.write("*No cross-scene negative pairs evaluated.*\n\n")

        # Section 4: Aggregate Metrics & Mandatory Caveats
        f.write("## 4. Overall Combined Metrics (With Mandatory Caveats)\n\n")
        f.write("> [!WARNING]\n")
        f.write("> **DO NOT cite this aggregate score as a generalized real-world accuracy benchmark.**  \n")
        f.write("> The benchmark composition contains limited controlled evidence (1 genuine unconstrained camera pair, 10 controlled transformations of a single photograph (`real_src_0001`), and 24 cross-scene negative pairs). Blending these categories into a single summary statistic masks whether the model performs on unconstrained captures versus synthetic distortions of one scene.\n\n")
        f.write(f"- **Combined Precision:** {all_m['prec']:.4f}\n")
        f.write(f"- **Combined Recall:** {all_m['rec']:.4f}\n")
        f.write(f"- **Combined F1-Score:** {all_m['f1']:.4f}\n")
        f.write(f"- **Combined Accuracy:** {all_m['acc']:.4f}\n")
        f.write(f"- **Combined FPR:** {all_m['fpr']:.4f}\n")
        if roc_auc_val is not None:
            f.write(f"- **Combined ROC-AUC:** {roc_auc_val:.4f}\n")
        f.write(f"- **Brier Score:** {brier:.4f}\n")
        f.write(f"- **Expected Calibration Error (ECE, 10 bins):** {ece:.4f}\n")
        f.write(f"- **Confusion Matrix:** TN={all_m['tn']}, FP={all_m['fp']}, FN={all_m['fn']}, TP={all_m['tp']}\n\n")

        # Section 5: Multi-Label Transformation Diagnostics
        if macro_f1 is not None:
            f.write("## 5. Multi-Label Transformation Performance\n\n")
            f.write(f"- **Macro-F1 (Supported Classes with Support > 0):** {macro_f1:.4f}\n\n")
            f.write("| Transformation Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|\n")
            for c, s in tfm_stats.items():
                f.write(f"| `{c}` | {s['precision']} | {s['recall']} | {s['f1']} | {s['support']} |\n")
            f.write("\n")

        # Section 6: Limitations & Methodology Disclosures
        f.write("## 6. Benchmark Methodology, Arbitration, & Disclosures\n\n")
        f.write("1. **Forensic Arbitration Rule Disclosure:** Relationship, direction, and transformation outputs may be affected by post-inference rule arbitration (e.g., container/sub-region checks for cropping, resolution parity checks). Final direction and transformation decisions reflect both neural MLP inference and rule-based physical constraints.\n")
        f.write("2. **Uncalibrated Model Confidence:** Raw confidence scores are sigmoid outputs from trained MLPs without temperature scaling, Platt scaling, or isotonic calibration.\n")
        f.write("3. **Single Source for Controlled Transforms:** All 10 controlled positive pairs originate from a single photograph (`real_src_0001`), so transform robustness may not generalize to other sensor resolutions, lighting conditions, or noise floors.\n")
        f.write("4. **Limited Independent Real Positives:** Only 1 contributor-confirmed unconstrained positive pair (`pair_0001`) is currently available in the benchmark.\n")
        f.write("5. **Strict Metric Separation:** These real-world results must never be averaged with synthetic held-out test metrics (`SIPGD_v2`).\n")

    print(f"[RealWorldEval] Report written to: {output_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
