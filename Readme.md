# Rete reproduction package

This repository accompanies [*Rete: Learning Namespace Representation for Program Repair*](https://mechtaev.com/files/icse23.pdf), ICSE 2023, by Nikhil Parasaram, Earl T. Barr, and Sergey Mechtaev. The paper evaluates variable ranking with CDU chains in three repair configurations: plastic surgery, Prophet, and Trident. Its evaluation uses 107 BugsInPy bugs, 35 ManyBugs bugs, and 28 Python programs for representation experiments.

## What this checkout can reproduce

| Component | Contents in this checkout | Current limit |
| --- | --- | --- |
| C/C++ feature extraction | Clang-based extractor and a batch driver | Requires LLVM/Clang 3.8.1 to build |
| Python Prophet | Feature extraction and feature-vector code | Training data and environment are not packaged |
| Trident and Rete search | Runtime, synthesis code, components, nine small examples, and repaired template/variable ordering | Original CodeBERT weights and probability outputs are absent |
| ManyBugs evaluation | One Coreutils example | The other cases and complete results are absent |
| BG107 timing figure | `Scripts/time_data.json` with 107 timings per tool | Raw patch and test outputs are absent |
| CoCoNut baseline | Adapted scripts | Its `fairseq-context` forks and models are absent |

Run `python3 Scripts/check_package.py` for an inventory. `--json` gives a machine-readable report; `--strict` exits nonzero until the paper's artifacts are complete. This is an audit, not a claim that the historical toolchains have been rebuilt.

The dataset ID notes also have unresolved gaps. `Dataset-Information/bg107_info.txt` lists 107 entries but only 106 distinct project/bug pairs because Luigi bug 10 occurs twice. `Dataset-Information/mb37_info.txt` lists 34 pairs, while the paper reports an MB35 evaluation subset. The missing IDs cannot be inferred from the archived files, so neither list should be used as an exact benchmark manifest yet.

## Layout

- `rete-feature-extracter/`: Clang extractor, Python learning code, and Prophet feature code. The directory name intentionally retains the original spelling.
- `Rete-Trident/`: Trident runtime, synthesizer, components, and small examples.
- `eval/coreutils_test/`: one archived ManyBugs-style case with KLEE outputs.
- `Dataset-Information/`: historical corpus notes, with the discrepancies above.
- `Scripts/`: timing data, plot generator, and package audit.
- `coconut/`: adapted CoCoNut scripts, without the required fairseq forks.

## Feature extraction

The extractor's CMake option is `RETE_LLVM`. With LLVM/Clang 3.8.1 installed at `/llvm-3.8.1`:

```sh
cmake -S rete-feature-extracter -B rete-feature-extracter/build -DRETE_LLVM=/llvm-3.8.1
cmake --build rete-feature-extracter/build
```

For a project with `compile_commands.json`, use its compilation database so the extractor receives the project's include paths and defines:

```sh
python3 rete-feature-extracter/rete_runner.py \
  --compile-commands /path/to/project/compile_commands.json \
  --executable rete-feature-extracter/build/tools/rete \
  --output-dir /path/to/output/chain_data
```

For simple sources, pass a source directory instead of `--compile-commands`. The driver scans C/C++ files recursively, preserves relative paths in the output, resumes valid outputs, and exits nonzero if any file fails. `--features` requests feature JSON instead of CDU-chain JSON; `--force` replaces existing outputs. The Wireshark wrapper retains the historical `capsa` filter and accepts the same overrides.

The historical container build is:

```sh
docker build -t rete/ubuntu-16.04-llvm-3.8.1 rete-feature-extracter/infra
docker build -t rete-feature rete-feature-extracter
```

It depends on Ubuntu 16.04 and LLVM 3.8.1 and has not been verified in a current Docker environment. The included build scripts are research-era sources rather than pinned modern images.

## Trident and Rete ranking

The root `Dockerfile` is the historical KLEE/Trident environment. Its source path and a malformed continuation have been corrected. If it builds in your environment, start with the small example:

```sh
docker build -t rete-trident .
docker run --rm -it rete-trident bash
cd /home/Trident/Rete-Trident
./tests/assignment/run
```

The Docker build was not verified here because Docker is unavailable. The container uses old Ubuntu, LLVM, KLEE, and Python versions, so its dependency repositories may need archival work. `Rete-Trident/README.md` documents the synthesizer's SMT interface.

The Rete enumerator now starts from at most 20 single-holed donor statements, explores template edits by distance, and lazily ranks concrete patches with the paper's score `distance + theta * mean(1/probability)`. It uses the paper's default `theta = 0.073` and at most 30 variables per hole. `--template-budget` can optionally limit the total search for experiments. The verified synthesis path and JSON patch serialization have also been repaired.

`--templates` requires `--model` pointing to a JSON export of variable probabilities. The adapter in `Rete-Trident/main/rankers.py` accepts a `default` mapping and optional `contexts` keyed by template code or `template_code@hole/path`. See `Rete-Trident/tests/probabilities.example.json` for **synthetic test values only**. A default mapping allows search to score newly generated templates; it is not a substitute for the paper's fine-tuned CodeBERT ranker. The original trained weights and per-context probabilities are missing, so the paper's ranking quality and repair counts cannot yet be reproduced. `--all` prints every generated patch, and `--theta` accepts fractional values.

The core algorithm checks can be run in an isolated environment:

```sh
python3 -m venv /tmp/rete-test-env
/tmp/rete-test-env/bin/pip install -r Rete-Trident/requirements-test.txt
/tmp/rete-test-env/bin/python -m unittest discover -s Rete-Trident/tests -p 'test_rete_algorithm.py'
```

## Python timing figure

The stored BG107 timing series can be processed without a plotting library:

```sh
python3 Scripts/plot.py --output bg107-curves.json
```

To render a figure, install `matplotlib` in your own environment and use `--output bg107-curves.png` or `.pdf`. The script reads its data relative to itself, checks the 107-row shape, and reproduces the archived 10-permutation mean curves. It does not regenerate timings from repair runs.

## Remaining recovery work

A faithful rerun of the paper still needs the original or retrained CodeBERT variable ranker and weights, its training helper, a corrected corpus manifest, the 28 training program snapshots, all 107 BugsInPy and 35 ManyBugs cases with test splits, and the baseline environments and raw patches. The paper describes a 20/80 test split with at least one failing test in the smaller part, but this checkout does not contain the per-bug split assignments. Reconstructing those choices from the published counts alone would change the experiment.
