"""Checks for the paper's patch score and the repaired synthesis path."""

import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main"))

from rete_priority import joint_score, ranked_bindings  # noqa: E402
import synthesis  # noqa: E402


class FixedRanker:
    def probabilities(self, code, hole_path, names):
        return {name: {"x": 0.8, "y": 0.2}.get(name, 0.01) for name in names}


class ReteAlgorithmTest(unittest.TestCase):
    def test_runtime_location_ids_accept_l_prefix(self):
        from pysmt.shortcuts import Symbol
        from pysmt.typing import ArrayType, BV32, BV8
        symbol = Symbol("choice!angelic!i32!L9!0", ArrayType(BV32, BV8))
        self.assertEqual(synthesis.RuntimeSymbol.parse(symbol).lid, "L9")

    def test_joint_score_is_paper_equation_one(self):
        self.assertAlmostEqual(joint_score(2, [0.5, 0.25]),
                               2 + 0.073 * (2 + 4) / 2)
        with self.assertRaises(ValueError):
            joint_score(0, [0])

    def test_bindings_are_lazy_and_ordered(self):
        options = [[("x", 0.8), ("y", 0.2)],
                   [("x", 0.8), ("y", 0.2)]]
        ranked = list(ranked_bindings(options, 1))
        self.assertEqual(len(ranked), 4)
        self.assertEqual(ranked[0][1], ("x", "x"))
        self.assertEqual(ranked[-1][1], ("y", "y"))
        self.assertEqual([score for score, _ in ranked],
                         sorted(score for score, _ in ranked))

    def test_probability_export_uses_context_then_default(self):
        from rankers import ChainRanker
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "probabilities.json"
            model.write_text(json.dumps({
                "default": {"x": 0.8, "y": 0.2},
                "contexts": {"(x + hole)@right": {"y": 0.9, "x": 0.1}}
            }), encoding="utf-8")
            ranker = ChainRanker(model)
            self.assertEqual(ranker.probabilities("(x + hole)", ("right",),
                                                   ["x", "y"])["y"], 0.9)
            self.assertEqual(ranker.probabilities("hole", (), ["x"])["x"], 0.8)

    def test_template_search_produces_concrete_typed_patches(self):
        addition = ROOT / "components/addition.smt2"
        components = [synthesis.make_component("x"),
                      synthesis.make_component("y")]
        components += synthesis.load_components([addition])
        by_name = dict(components)
        donor = (("addition", by_name["addition"]), {
            "left": (components[0], {}), "right": (components[1], {})})
        args = SimpleNamespace(theta=None, model=None, template_budget=1000)
        enumerator = synthesis.ReteEnumerator([(donor, {})], components, args,
                                               ranker=FixedRanker())
        patches = list(itertools.islice(enumerator.enumerate_templates(
            components, 3, synthesis.TridentType.I32, False, True), 10))
        codes = [synthesis.program_to_code((tree, {})) for tree in patches]
        self.assertIn("(x + y)", codes)
        self.assertIn("x", codes)
        self.assertTrue(all("hole" not in code for code in codes))
        self.assertTrue(all(enumerator._depth(tree) <= 3 for tree in patches))

    def test_twenty_starting_templates_do_not_cap_graph_search(self):
        addition = synthesis.load_components(
            [ROOT / "components/addition.smt2"])[0]
        variables = [synthesis.make_component(f"x{i}") for i in range(20)]
        hole = (("hole", None), {})
        donors = [((addition, {"left": hole, "right": (variable, {})}), {})
                  for variable in variables]

        class UniformRanker:
            def probabilities(self, code, hole_path, names):
                return {name: 0.8 for name in names}

        enumerator = synthesis.ReteEnumerator(
            donors, [addition] + variables,
            SimpleNamespace(theta=None, model=None), ranker=UniformRanker())
        expanded = []
        original = enumerator._instantiations

        def track(tree, distance, requirement):
            expanded.append(enumerator._signature(tree))
            yield from original(tree, distance, requirement)

        with patch.object(enumerator, "_instantiations", side_effect=track):
            next(enumerator.enumerate_templates(
                [addition] + variables, 2, synthesis.TridentType.I32,
                False, True))
        self.assertGreater(len(set(expanded)), 20)

    def test_program_json_keeps_every_child(self):
        addition = synthesis.load_components([ROOT / "components/addition.smt2"])[0]
        left = synthesis.make_component("x")
        right = synthesis.make_component("y")
        program = ((addition, {"left": (left, {}), "right": (right, {})}), {})
        encoded = synthesis.program_to_json(program)
        self.assertEqual(set(encoded["tree"]["children"]), {"left", "right"})
        decoded = synthesis.program_of_json(encoded, [addition, left, right])
        self.assertEqual(synthesis.program_to_code(decoded), "(x + y)")

    def test_synthesize_returns_verified_tree(self):
        variable = synthesis.make_component("x")
        tree = (variable, {})
        specification = {"test": ([object()], object())}
        def enumerate_one(*args, **kwargs):
            yield tree
        with patch.object(synthesis, "extract_lids",
                          return_value=({"L1": synthesis.TridentType.I32}, False)), \
             patch.object(synthesis, "verify",
                          return_value=synthesis.VerificationSuccess({})):
            result = next(synthesis.synthesize([variable], 1, specification,
                                               1, enumerate_one))
        self.assertEqual(result["L1"][0], tree)

    def test_ranked_synthesis_checks_a_real_smt_specification(self):
        from pysmt.shortcuts import And, BV, Equals, Symbol
        from pysmt.typing import BV32
        memory = synthesis.Klee.memory_type
        angelic = Symbol("choice!angelic!i32!L9!0", memory)
        variable = Symbol("choice!rvalue!L9!0!x", memory)
        output = Symbol("output!i32!output!0", memory)
        value = lambda array: synthesis.Klee.interpret_memory(
            array, synthesis.TridentType.I32)
        path = And(Equals(value(variable), BV(2, 32)),
                   Equals(value(angelic), value(output)))
        assertion = Equals(Symbol("output!0", BV32), BV(2, 32))
        specification = {"t": ([path], assertion)}
        components = synthesis.get_components(specification)
        donor = (components[0], {})
        enumerator = synthesis.ReteEnumerator(
            [(donor, {})], components,
            SimpleNamespace(theta=None, model=None, template_budget=100),
            ranker=FixedRanker())
        repaired = next(synthesis.synthesize(components, 1, specification, 1,
                                            enumerator.enumerate_templates))
        self.assertEqual(synthesis.program_to_code(repaired["L9"]), "x")


if __name__ == "__main__":
    unittest.main()
