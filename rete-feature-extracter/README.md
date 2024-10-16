# Rete C/C++ feature extractor

This is the Clang-based feature and CDU-chain extractor used by the Rete package. It requires LLVM and Clang 3.8.1. From the repository root, build it with:

```sh
cmake -S rete-feature-extracter -B rete-feature-extracter/build -DRETE_LLVM=/llvm-3.8.1
cmake --build rete-feature-extracter/build
```

The legacy two-stage Docker build is documented in the [root README](../Readme.md). It has not been verified on a current host.

To process a source tree with a Clang compilation database:

```sh
python3 rete-feature-extracter/rete_runner.py \
  --compile-commands /path/to/compile_commands.json \
  --executable rete-feature-extracter/build/tools/rete \
  --output-dir /path/to/chain_data
```

For self-contained C/C++ files, use `rete_runner.py /path/to/sources` instead of `--compile-commands`. `--features` extracts feature JSON; the default extracts CDU-chain JSON. `--force` replaces valid existing outputs, `--jobs` sets the parallel worker count, and `--contains` filters source paths. The driver reports failures and returns a nonzero status if any extraction fails.

For one source file, the underlying executable accepts `-get-chain-data -output=/path/to/data.json` after the source path. Omit `-get-chain-data` to extract feature JSON. Pass `-p /path/to/build-directory` when the source requires a compilation database.
