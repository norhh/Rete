from unittest.case import TestCase

from pysmt.shortcuts import get_model, Symbol, BV, Equals, And, Or
from synthesis import encode, decode
from pysmt.typing import BOOL, BV32

class SynthesisTest(TestCase):

    def test_single_component(self):
        lid = "f"
        tid = "t"
        var_name = "x"
        component = Symbol(f"rvalue_{var_name}", BV32)
        output = Symbol("output!i32!output!0", BV32)
        angelic = Symbol(f"choice!angelic!i32!{lid}!0", BV32)
        expected = BV(2, 32)
        var = Symbol(f"choice!rvalue!i32!{lid}!0!{var_name}", BV32)
        assertion = Equals(output, expected)
        path = And(Equals(var, expected),
                   Equals(angelic, output))
        spec = { tid: ([path], assertion)}
        formula = encode([lid], [component], 1, spec)
        model = get_model(formula)
        result = decode([lid], [component], model)
        print("foo")
        self.assertEqual(result, (component, {}))

if __name__ == '__main__':
    unittest.main()
