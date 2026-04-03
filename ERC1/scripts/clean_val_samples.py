#!/usr/bin/env python
"""
Clean debiased validation samples by removing obviously corrupted entries.

Current default rule:
- drop samples whose `target_utterance` token count exceeds `max_target_words`
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def word_count(text: str) -> int:
    return len(str(text).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "json" / "val_samples_debiased.json"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "json" / "val_samples_debiased_clean.json"),
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "json" / "val_samples_debiased_clean_report.json"),
    )
    parser.add_argument("--max_target_words", type=int, default=128)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    data = json.loads(input_path.read_text(encoding="utf-8"))

    kept = []
    removed = []

    for row in data:
        target = str(row.get("target_utterance", ""))
        target_words = word_count(target)
        if target_words > args.max_target_words:
            removed.append(
                {
                    "sample_id": row.get("sample_id"),
                    "dataset": row.get("dataset"),
                    "emotion": row.get("emotion"),
                    "target_words": target_words,
                    "target_preview": target[:180].replace("\n", " "),
                    "reason": f"target_words>{args.max_target_words}",
                }
            )
            continue
        kept.append(row)

    output_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    before_counter = Counter(x.get("emotion") for x in data)
    after_counter = Counter(x.get("emotion") for x in kept)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "max_target_words": args.max_target_words,
        "total_before": len(data),
        "total_after": len(kept),
        "removed_count": len(removed),
        "removed_ratio": (len(removed) / len(data)) if data else 0.0,
        "label_distribution_before": dict(before_counter),
        "label_distribution_after": dict(after_counter),
        "removed_samples": removed,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded: {len(data)}")
    print(f"Removed: {len(removed)}")
    print(f"Saved cleaned file: {output_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
