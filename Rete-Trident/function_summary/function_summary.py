from sys import argv
from typing import List
import os
import pysmt.fnode
from pysmt.fnode import FNode
from pysmt.solvers.solver import Model
from pysmt.substituter import MGSubstituter
from pysmt.smtlib.parser import SmtLibParser
from pysmt.shortcuts import (
    is_sat,
    get_model,
    Symbol,
    BV,
    Equals,
    EqualsOrIff,
    And,
    Or,
    TRUE,
    FALSE,
    Select,
    BVConcat,
    simplify,
    FreshSymbol,
    write_smtlib
)
import pysmt.environment
from pysmt.walkers import DagWalker, IdentityDagWalker
from pysmt.environment import get_env
from pysmt.smtlib.printers import SmtPrinter, SmtDagPrinter
from pysmt.typing import BOOL, BV32, BV8, ArrayType
import pysmt.operators as op
from pysmt.printers import HRPrinter


def get_path_constraints(path: str) -> List[FNode]:
    smt_files: List[str] = []
    for file in os.listdir(argv[1]):
        if ".smt2" in file:
            if file.split(".")[0] + "model.err" in os.listdir(argv[1]) or \
                    file.split(".")[0] + "ptr.err" in os.listdir(argv[1]):
                continue
            smt_files.append(os.path.join(argv[1], file))
    path_constraints: List[FNode] = []
    for file in smt_files:
        with open(file) as f:
            parser = SmtLibParser()
            script = parser.get_script(f)
            path_constraints.append(script.get_last_formula())
    return path_constraints


def remove_model_var(path_constraints: List[FNode]):
    new_constraints: List[FNode] = []
    for constraint in path_constraints:
        substituter = MGSubstituter(get_env())
        subs = {}
        for var in constraint.get_free_variables():
            if "model_version" == str(var):
                for i in range(1, 4):
                    lhs = Select(var, BV(i, 32))
                    subs[lhs] = BV(0, 8)
                subs[Select(var, BV(0, 32))] = BV(1, 8)
        new_constraint: FNode = substituter.substitute(constraint, subs=subs)
        new_constraints.append(new_constraint)

    return new_constraints


def get_integer_chunk(variable: FNode) -> FNode:
    return BVConcat(*[Select(variable, BV(i, 32)) for i in range(3, -1, -1)])


def combine_path_constraints(path_constraints: List[FNode]) -> FNode:
    combined_constraint: FNode = Or(*path_constraints)
    for var in combined_constraint.get_free_variables():
        if "DONE" not in var.serialize():
            new_var = Symbol("rhole_" + str(var), BV32)
        else:
            new_var = Symbol("lhole_" + str(var), BV32)
        combined_constraint = And(
            combined_constraint, Equals(new_var, get_integer_chunk(var))
        )

    return simplify(combined_constraint)


def main():
    path = argv[1]
    path_constraints = get_path_constraints(path)
    path_constraints = remove_model_var(path_constraints)
    path_constraint = combine_path_constraints(path_constraints)
    rreturn = Symbol("rreturn", BV32)
    path_constraint = And(path_constraint, Equals(rreturn, BV(0, 32)))
    write_smtlib(path_constraint, os.path.join(os.getcwd(), "summaries/{}".format(argv[2])))


if __name__ == "__main__":
    main()
