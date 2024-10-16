"""Audit which parts of the ICSE 2023 Rete reproduction are in this checkout."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CODE = {
    "C/C++ feature extractor": "rete-feature-extracter/tools/rete.cpp",
    "Trident synthesizer": "Rete-Trident/main/synthesis.py",
    "Trident runtime": "Rete-Trident/runtime/trident_runtime.c",
    "Python Prophet feature extractor": "rete-feature-extracter/learning/prophet.py",
    "sample ManyBugs case": "eval/coreutils_test/test_c160afe/run.sh",
}
MISSING_RESEARCH_ARTIFACTS = {
    "Trained Rete variable probabilities": "Rete-Trident/models/probabilities.json",
    "CDU/DU training helper": "rete-feature-extracter/learning/model_trainers.py",
    "CoCoNut fairseq fork": "coconut/fairseq-context",
    "CoCoNut preprocessing fork": "coconut/fairseq-context_good",
}


def bug_ids(path):
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        project, ids = line.split(":", 1)
        result.extend((project.strip().lower(), int(number))
                      for number in re.findall(r"\d+", ids))
    return result


def manybugs_ids(path):
    return re.findall(r"(?m)^[0-9a-f]+-[0-9a-f]+\s*$",
                      path.read_text(encoding="utf-8"))


def audit(root=ROOT):
    files = {label: (root / relative).exists()
             for label, relative in REQUIRED_CODE.items()}
    files.update({label: (root / relative).exists()
                  for label, relative in MISSING_RESEARCH_ARTIFACTS.items()})
    bg = bug_ids(root / "Dataset-Information/bg107_info.txt")
    duplicates = {f"{project}:{number}": count
                  for (project, number), count in Counter(bg).items() if count > 1}
    mb = manybugs_ids(root / "Dataset-Information/mb37_info.txt")
    with (root / "Scripts/time_data.json").open(encoding="utf-8") as stream:
        times = json.load(stream)
    time_counts = {name: len(values) for name, values in times.items()}
    return {
        "files": files,
        "benchmarks": {
            "paper_bugsinpy_count": 107,
            "listed_bugsinpy_entries": len(bg),
            "unique_bugsinpy_ids": len(set(bg)),
            "duplicate_bugsinpy_ids": duplicates,
            "paper_manybugs_count": 35,
            "listed_manybugs_pairs": len(mb),
            "bundled_manybugs_case_directories": len(list((root / "eval").glob("*_test/test_*"))),
            "timing_rows": time_counts,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable report")
    parser.add_argument("--strict", action="store_true",
                        help="exit nonzero while paper artifacts are missing")
    args = parser.parse_args(argv)
    report = audit()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for label, present in report["files"].items():
            print(f"{'PRESENT' if present else 'MISSING'}  {label}")
        for label, value in report["benchmarks"].items():
            print(f"{label}: {value}")
    incomplete = (not all(report["files"].values())
                  or report["benchmarks"]["unique_bugsinpy_ids"] != 107
                  or report["benchmarks"]["listed_manybugs_pairs"] < 35
                  or report["benchmarks"]["bundled_manybugs_case_directories"] < 35
                  or any(count != 107 for count in report["benchmarks"]["timing_rows"].values()))
    return int(args.strict and incomplete)


if __name__ == "__main__":
    raise SystemExit(main())
