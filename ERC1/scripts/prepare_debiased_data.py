#!/usr/bin/env python
"""
Prepare debiased train/validation files for CoT training/evaluation.

Actions:
1. Remove prev_emotion / prev_speaker leakage fields.
2. For Empathetic multi-turn dialogue clusters, keep only the last turn.
3. Preserve all single-turn / non-dialogue samples such as GoEmotions.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def empathetic_group_key(sample: dict) -> str:
    return str(sample["sample_id"]).rsplit("_", 1)[0]


def empathetic_turn_index(sample: dict) -> int:
    return int(str(sample["sample_id"]).rsplit("_", 1)[1])


def debias_samples(samples: list[dict]) -> tuple[list[dict], dict]:
    kept = []
    empathetic_groups = defaultdict(list)

    for sample in samples:
        if sample.get("dataset") == "empathetic" and "_conv:" in str(sample.get("sample_id", "")):
            empathetic_groups[empathetic_group_key(sample)].append(sample)
        else:
            cleaned = dict(sample)
            cleaned["prev_emotion"] = None
            cleaned["prev_speaker"] = None
            kept.append(cleaned)

    multi_turn_groups = 0
    for _, items in empathetic_groups.items():
        items.sort(key=empathetic_turn_index)
        if len(items) > 1:
            multi_turn_groups += 1
        cleaned = dict(items[-1])
        cleaned["prev_emotion"] = None
        cleaned["prev_speaker"] = None
        kept.append(cleaned)

    kept.sort(key=lambda x: str(x["sample_id"]))
    report = {
        "input_count": len(samples),
        "output_count": len(kept),
        "removed_count": len(samples) - len(kept),
        "dataset_before": dict(Counter(s["dataset"] for s in samples)),
        "dataset_after": dict(Counter(s["dataset"] for s in kept)),
        "multi_turn_empathetic_groups": multi_turn_groups,
    }
    return kept, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    args = parser.parse_args()

    train_samples = load_json(args.train_input)
    train_debiased, train_report = debias_samples(train_samples)
    save_json(args.train_output, train_debiased)

    val_samples = load_json(args.val_input)
    val_debiased, val_report = debias_samples(val_samples)
    save_json(args.val_output, val_debiased)

    print("Train report:")
    print(json.dumps(train_report, ensure_ascii=False, indent=2))
    print("Validation report:")
    print(json.dumps(val_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
