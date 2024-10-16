"""End-to-end checks for the dependency-free extraction driver."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rete_runner import main  # noqa: E402


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.executable = self.root / "fake rete"
        self.executable.write_text(
            f"#!{sys.executable}\n"
            "import json, pathlib, sys\n"
            "source = pathlib.Path(sys.argv[1])\n"
            "if source.name == 'fail.c':\n"
            "    print('forced failure', file=sys.stderr)\n"
            "    sys.exit(2)\n"
            "output = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('-output='))\n"
            "pathlib.Path(output).write_text(json.dumps({'source': str(source), 'args': sys.argv[2:]}))\n",
            encoding="utf-8")
        self.executable.chmod(0o755)
        self.out = self.root / "outputs"

    def run_driver(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main([*arguments, "--executable", str(self.executable),
                           "--output-dir", str(self.out), "--jobs", "2"])
        return status, out.getvalue(), err.getvalue()

    def test_recursive_sources_keep_distinct_output_paths_and_resume(self):
        for directory in ("a", "b"):
            source = self.root / "src" / directory / "same.c"
            source.parent.mkdir(parents=True)
            source.write_text("int x;\n", encoding="utf-8")
        status, message, _ = self.run_driver(str(self.root / "src"))
        self.assertEqual(status, 0)
        self.assertIn("2 written", message)
        self.assertTrue((self.out / "a/same.c.json").is_file())
        self.assertTrue((self.out / "b/same.c.json").is_file())
        status, message, _ = self.run_driver(str(self.root / "src"))
        self.assertEqual(status, 0)
        self.assertIn("2 skipped", message)

    def test_failure_is_reported_and_does_not_leave_partial_output(self):
        source = self.root / "src" / "fail.c"
        source.parent.mkdir()
        source.write_text("int x;\n", encoding="utf-8")
        status, _, errors = self.run_driver(str(source.parent))
        self.assertEqual(status, 1)
        self.assertIn("forced failure", errors)
        self.assertFalse((self.out / "fail.c.json").exists())

    def test_compile_database_passes_build_directory(self):
        source = self.root / "code.c"
        source.write_text("int x;\n", encoding="utf-8")
        database = self.root / "compile_commands.json"
        database.write_text(json.dumps([{
            "directory": str(self.root), "file": "code.c", "command": "cc -c code.c"
        }]), encoding="utf-8")
        status, _, _ = self.run_driver("--compile-commands", str(database))
        self.assertEqual(status, 0)
        output = json.loads((self.out / "code.c.json").read_text(encoding="utf-8"))
        self.assertEqual(output["args"][-2:], ["-p", str(self.root)])


if __name__ == "__main__":
    unittest.main()
