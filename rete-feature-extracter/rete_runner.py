"""Extract Rete features from C/C++ files without losing per-file failures.

The C++ executable uses Clang's compilation database. Pass --compile-commands
for projects whose sources need project-specific include paths or defines.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
HERE = Path(__file__).resolve().parent


def sources_from_directory(directory):
    return sorted(path.resolve() for path in directory.rglob("*")
                  if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES)


def sources_from_database(database):
    with database.open(encoding="utf-8") as stream:
        entries = json.load(stream)
    if not isinstance(entries, list):
        raise ValueError("compilation database must be a JSON array")
    paths = set()
    for entry in entries:
        source = (database.parent / entry["directory"] / entry["file"]).resolve()
        if source.suffix.lower() in SOURCE_SUFFIXES:
            paths.add(source)
    return sorted(paths)


def output_for(source, base, output_dir):
    try:
        relative = source.relative_to(base)
    except ValueError:
        digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        relative = Path(digest) / source.name
    return output_dir / (str(relative) + ".json")


def extract(source, base, output_dir, executable, database, chain_data, force):
    destination = output_for(source, base, output_dir)
    if destination.exists() and not force:
        try:
            with destination.open(encoding="utf-8") as stream:
                json.load(stream)
            return source, "skipped", ""
        except (OSError, ValueError):
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".rete-", suffix=".json", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    command = [str(executable), str(source), f"-output={temporary}"]
    if chain_data:
        command.append("-get-chain-data")
    if database is not None:
        command.extend(["-p", str(database.parent)])
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   check=False)
        if completed.returncode:
            return source, "failed", completed.stderr.strip() or f"exit {completed.returncode}"
        with temporary.open(encoding="utf-8") as stream:
            json.load(stream)
        temporary.replace(destination)
        return source, "written", ""
    except (OSError, ValueError) as error:
        return source, "failed", str(error)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("source_dir", nargs="?", type=Path,
                        help="recursively scan a source directory")
    inputs.add_argument("--compile-commands", type=Path, metavar="JSON",
                        help="use a Clang compilation database")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/chain_data"))
    parser.add_argument("--executable", type=Path, default=HERE / "build/tools/rete")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--contains", default="", help="select paths containing this text")
    parser.add_argument("--features", action="store_true",
                        help="extract feature vectors instead of CDU chains")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    executable = args.executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error(f"extractor is unavailable or not executable: {executable}")
    database = args.compile_commands.resolve() if args.compile_commands else None
    if database:
        if not database.is_file():
            parser.error(f"compilation database does not exist: {database}")
        try:
            sources = sources_from_database(database)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"invalid compilation database: {error}")
        base = database.parent
    else:
        base = args.source_dir.resolve()
        if not base.is_dir():
            parser.error(f"source directory does not exist: {base}")
        sources = sources_from_directory(base)
    sources = [source for source in sources if args.contains in str(source)]
    if not sources:
        parser.error("no C/C++ source files matched")
    counts = {"written": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(extract, source, base, args.output_dir.resolve(),
                               executable, database, not args.features, args.force)
                   for source in sources]
        for future in as_completed(futures):
            source, status, detail = future.result()
            counts[status] += 1
            if status == "failed":
                print(f"FAILED {source}: {detail}", file=sys.stderr)
    print(f"Rete extraction: {counts['written']} written, "
          f"{counts['skipped']} skipped, {counts['failed']} failed")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
