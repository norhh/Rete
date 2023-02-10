import argparse
from sys import stdout
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Union, Optional, NamedTuple
from enum import Enum, auto
import re
from random import randint
import json
from collections import namedtuple
from random import sample
import pysmt.fnode
from pysmt.solvers.solver import Model
from pysmt.smtlib.parser import SmtLibParser
from pysmt.shortcuts import is_sat, get_model, Symbol, BV, Equals, EqualsOrIff, And, Or, TRUE, FALSE, Select, BVConcat, simplify
import pysmt.environment
from pysmt.walkers import DagWalker, IdentityDagWalker
from pysmt.environment import get_env
from pysmt.smtlib.printers import SmtPrinter, SmtDagPrinter
from pysmt.typing import BOOL, BV32, BV8, ArrayType
import pysmt.operators as op
from time import time
from funcy import all_fn, any_fn, complement
from functools import lru_cache
from copy import deepcopy, copy
import io
import heapq
import math
logger = logging.getLogger(__name__)
no_ranker = False

try:
    from rankers import ChainRanker
    ChainRanker().has_models()
except Exception:
    no_ranker = True
    logger.info("No ranker info found, using default enumeration.")

Formula = Union[pysmt.fnode.FNode]

lvalue_component = "(declare-const rvalue_{} (_ BitVec 32))(declare-const lvalue_{} (_ BitVec 32))(declare-const rreturn (_ BitVec 32))(declare-const lreturn (_ BitVec 32))(assert (and (= rreturn rvalue_{}) (= lreturn lvalue_{})))"
rvalue_component = "(declare-const rvalue_{} (_ BitVec 32))(declare-const rreturn (_ BitVec 32))(assert (= rreturn rvalue_{}))"

COMPONENTS = []



def collect_symbols(formula, predicate):

    class SymbolCollector(DagWalker):
        def __init__(self):
            DagWalker.__init__(self, env=get_env(), invalidate_memoization=True)
            self.symbols = set()
            self.functions = {}
            for o in op.all_types():
                if o == op.SYMBOL:
                    self.functions[o] = self.walk_symbol
                else:
                    self.functions[o] = self.walk_none

        def _get_key(self, formula, **kwargs):
            return formula

        def walk_symbol(self, formula, args, **kwargs):
            pred = kwargs['pred']
            if pred(formula):
                self.symbols.add(formula)

    collector = SymbolCollector()
    collector.walk(formula, pred=predicate)
    return collector.symbols


def dump(formula, file):
    with open(file, "w") as f:
        p = SmtPrinter(f)
        p.printer(formula)


# Variables may have the following instances:
class Instance(Enum):
    EXECUTION = auto()  # parameterized by index of execution along given path
    TEST      = auto()  # parameterized by test id
    LOCATION  = auto()  # parameterized by location id
    PATH      = auto()  # parameterized by path id
    COMPONENT = auto()  # parameterized by component id
    NODE      = auto()  # parameterized by encoding node index

    @staticmethod
    def of_symbol(symbol: Formula, parameter: Union[int, str], inst) -> Formula:
        prefix = { Instance.EXECUTION: f"exe:{parameter}!",
                   Instance.TEST:      f"test:{parameter}!",
                   Instance.LOCATION:  f"loc:{parameter}!",
                   Instance.PATH:      f"path:{parameter}!",
                   Instance.COMPONENT: f"comp:{parameter}!",
                   Instance.NODE:      f"node:{parameter}!" }
        return Symbol(prefix[inst] + symbol.symbol_name(), symbol.symbol_type())

    @staticmethod
    def of_formula(formula: Formula, parameter: Union[int, str], inst, predicate) -> Formula:
        return formula.substitute({
            s: Instance.of_symbol(s, parameter, inst) for s in collect_symbols(formula, predicate)
            })

    @staticmethod
    def _regex(s):
        return {
            Instance.EXECUTION: re.compile(r'exe:(\d+)!'),
            Instance.TEST:      re.compile(r'test:(\w+)!'),
            Instance.LOCATION:  re.compile(r'loc:(\w+)!'),
            Instance.PATH:      re.compile(r'path:(\w+)!'),
            Instance.COMPONENT: re.compile(r'comp:(\w+)!'),
            Instance.NODE:      re.compile(r'node:(\d+)!')
        }[s]

    @staticmethod
    def check(symbol: Formula) -> Set['Instance']:
        result = set()
        for i in Instance:
            if Instance._regex(i).search(symbol.symbol_name()):
                result.add(i)
        return result

    @staticmethod
    def parameter(symbol: Formula, inst) -> Union[int, str]:
        par = Instance._regex(inst).search(symbol.symbol_name()).group(0)
        if inst == Instance.EXECUTION or inst == Instance.NODE:
            return int(par)
        else:
            return par


# Component is a pair of component id and component semantics expressed as 
# formula over symbols representing program variables, holes, constants and return values
# There are some restriction for components:
#   - all lvalues should be defined in terms of rvalues for all inputs
Component = Tuple[str, Formula]

class TridentType(Enum):
    I32 = auto()   # signed integer
    BOOL = auto()  # boolean
    ARRAY = auto()
    def __str__(self):
        return {
            TridentType.BOOL: "bool",
            TridentType.I32:  "i32",
            TridentType.ARRAY: "array"
        }[self]

    @staticmethod
    def parse(s):
        return {
            "bool": TridentType.BOOL,
            "i32":  TridentType.I32,
            "array": TridentType.ARRAY,
        }[s]


class SymbolData(NamedTuple):
    lid: Optional[str] = None
    eid: Optional[int] = None
    name: Optional[str] = None
    type: Optional[TridentType] = None


class ComponentSymbol(Enum):
    CONST   = auto()    # constant
    RRETURN = auto()    # rvalue output
    LRETURN = auto()    # lvalue output
    RBRANCH = auto()    # rvalue of branch
    LBRANCH = auto()    # lvalue of branch
    RVALUE  = auto()    # rvalue of variable
    LVALUE  = auto()    # lvalue of variable
    RHOLE   = auto()    # rvalue of subexpression
    LHOLE   = auto()    # lvalue of subexpression
    TEMPORARY = auto()  # temporary variable in the component

    @staticmethod
    def _regex(s):
        return {
            ComponentSymbol.CONST:   re.compile(r'const_([\d\w.\[\]]+)$'),
            ComponentSymbol.RRETURN: re.compile(r'(rreturn)$'),    
            ComponentSymbol.LRETURN: re.compile(r'(lreturn)$'),    
            ComponentSymbol.RBRANCH: re.compile(r'(rbranch)$'),    
            ComponentSymbol.LBRANCH: re.compile(r'(lbranch)$'),    
            ComponentSymbol.RVALUE:  re.compile(r'rvalue_([\d\w.\[\]]+)$'),
            ComponentSymbol.LVALUE:  re.compile(r'lvalue_([\d\w.\[\]]+)$'),
            ComponentSymbol.RHOLE:   re.compile(r'rhole_([\d\w.\[\]]+)$'),
            ComponentSymbol.LHOLE:   re.compile(r'lhole_([\d\w.\[\]]+)$'),
            ComponentSymbol.TEMPORARY: re.compile(r'value_([\d\w.\[\]]+)$')
        }[s]

    @staticmethod
    def check(symbol: Formula) -> Optional['ComponentSymbol']:
        for s in ComponentSymbol:
            if ComponentSymbol._regex(s).search(symbol.symbol_name()):
                return s
        return None

    is_const      = lambda s: ComponentSymbol.check(s) == ComponentSymbol.CONST
    is_rvalue     = lambda s: ComponentSymbol.check(s) == ComponentSymbol.RVALUE
    is_lvalue     = lambda s: ComponentSymbol.check(s) == ComponentSymbol.LVALUE
    is_rhole      = lambda s: ComponentSymbol.check(s) == ComponentSymbol.RHOLE
    is_lhole      = lambda s: ComponentSymbol.check(s) == ComponentSymbol.LHOLE
    is_rreturn    = lambda s: ComponentSymbol.check(s) == ComponentSymbol.RRETURN
    is_lreturn    = lambda s: ComponentSymbol.check(s) == ComponentSymbol.LRETURN
    is_rbranch    = lambda s: ComponentSymbol.check(s) == ComponentSymbol.RBRANCH
    is_lbranch    = lambda s: ComponentSymbol.check(s) == ComponentSymbol.LBRANCH
    is_special    = lambda s: ComponentSymbol.check(s) != None
    is_nonspecial = lambda s: ComponentSymbol.check(s) == None
    is_temporary = lambda  s: ComponentSymbol.check(s) == ComponentSymbol.TEMPORARY

    @staticmethod
    def parse(symbol: Formula) -> SymbolData:
        s = ComponentSymbol.check(symbol)
        if s is None:
            n = str(symbol)
        else:
            n = ComponentSymbol._regex(s).search(symbol.symbol_name()).group(1)
        if symbol.symbol_type() == BOOL:
            return SymbolData(type=TridentType.BOOL, name=n)
        elif symbol.symbol_type() == BV32:
            return SymbolData(type=TridentType.I32, name=n)
        else:
            return SymbolData(type=TridentType.I32, name=n)  # TODO: Handle this better
        raise ValueError(f"unsupported symbol type {symbol.symbol_type}")

    @staticmethod
    def const(name) -> Formula:
        #TODO: support bool type?
        return Symbol(f"const_{name}", BV32)

    @staticmethod
    def branch_of(s: Formula) -> Formula:
        return Symbol(re.sub(r'return$', 'branch', s.symbol_name()), s.symbol_type())


class ComponentSemantics:
    @staticmethod
    def get_rreturn(formula):
        rreturns = list(collect_symbols(formula, ComponentSymbol.is_rreturn))
        return rreturns[0] if rreturns else None

    @staticmethod
    def get_lreturn(formula):
        lreturns = list(collect_symbols(formula, ComponentSymbol.is_lreturn))
        return lreturns[0] if lreturns else None

    @staticmethod
    def get_rhole(formula, name):
        rholes = list(collect_symbols(formula, all_fn(ComponentSymbol.is_rhole, lambda x: ComponentSymbol.parse(x).name == name)))
        return rholes[0] if rholes else None

    @staticmethod
    def get_lhole(formula, name):
        lholes = list(collect_symbols(formula, all_fn(ComponentSymbol.is_lhole, lambda x: ComponentSymbol.parse(x).name == name)))
        return lholes[0] if lholes else None

    @staticmethod
    def get_lvalue(formula, name):
        lvalues = list(collect_symbols(formula, all_fn(ComponentSymbol.is_lvalue, lambda x: ComponentSymbol.parse(x).name == name)))
        return lvalues[0] if lvalues else None


# Program is a pair of 
# - a tree of components
# - a valuation of parameters (constants)
# A tree of components is represented as pair of
# - a root component
# - a mapping from hole ids to subprograms
# There are some restrictions on the structure of programs:
# - program cannot assign the same lvalue twice
ComponentTree = Tuple[Component, Dict[str, 'ComponentTree']]

        

Program = Tuple[ComponentTree, Dict[str, int]]

def program_to_formula(program: Program) -> Formula:
    (tree, constants) = program

    def tree_to_formula(tree, node_counter):
        nid = node_counter[0]
        node_counter[0] += 1
        ((cid, root), children) = tree
        branch_substitution = {}
        rreturn = ComponentSemantics.get_rreturn(root)
        root_rbranch = None
        if rreturn:
            root_rbranch = Instance.of_symbol(ComponentSymbol.branch_of(rreturn), nid, Instance.NODE)
            branch_substitution[rreturn] = root_rbranch
        lreturn = ComponentSemantics.get_lreturn(root)
        root_lbranch = None
        if lreturn:
            root_lbranch = Instance.of_symbol(ComponentSymbol.branch_of(lreturn), nid, Instance.NODE)
            branch_substitution[lreturn] = root_lbranch
        branch_root = root.substitute(branch_substitution)
        is_return = any_fn(ComponentSymbol.is_rreturn, ComponentSymbol.is_lreturn)
        # Keeping constants and variables because they are global; holes because they are replaced with children
        instantiated_root = Instance.of_formula(branch_root, nid, Instance.NODE, is_return)
        hole_substitution = {}
        children_semantics = TRUE()
        for hole_id, subtree in children.items():
            (subtree_semantics, subtree_rbranch, subtree_lbranch) = tree_to_formula(subtree, node_counter)
            if subtree_rbranch:
                rhole = ComponentSemantics.get_rhole(root, hole_id)
                if rhole:
                    hole_substitution[rhole] = subtree_rbranch
            if subtree_lbranch:
                lhole = ComponentSemantics.get_lhole(root, hole_id)
                if lhole:
                    hole_substitution[lhole] = subtree_lbranch
            children_semantics = And(children_semantics, subtree_semantics)              
        bound_root = instantiated_root.substitute(hole_substitution)
        result = And(bound_root, children_semantics) if children else bound_root 
        return (result, root_rbranch, root_lbranch)

    (semantics, rbranch, _) = tree_to_formula(tree, [0])
    rreturn = ComponentSemantics.get_rreturn(tree[0][1])
    if rreturn:
        semantics = And(semantics, EqualsOrIff(rreturn, rbranch))
    const_substitution = { ComponentSymbol.const(k): BV(v, 32) for k, v in constants.items() }
    return semantics.substitute(const_substitution)

    

def program_to_code(program: Program) -> str:
    (tree, constants) = program

    unary_operators = {
        'post-increment': '++',
        'post-decrement': '--',
        'minus': '-',
        'logical-not': '!',
        'bitwise-not': '~',
    }
    binary_operators = {
        'assignment': '=',
        'addition': '+',
        'subtraction': '-',
        'multiplication': '*',
        'division': '/',
        'remainder': '%',
        'equal': '==',
        'not-equal': '!=',
        'less-than': '<',
        'less-or-equal': '<=',
        'greater-than': '>',
        'greater-or-equal': '>=',
        'logical-and': '&&',
        'logical-or': '||',
        'sequence': ';',
        'bitwise-and': '&',
    }

    def get_args_from_children(children) -> str:
        """
        Assumes function summaries have args of form value_arg1_a, value_arg2_b, ....
        where arg1, arg2, ... denotes the order of the argument to the function.
        Some variables are suffixed with _DONE which are used to get path constraints from klee
        :param children:
        :return:
        """
        args = []
        for i in range(1, 10):    # Assume atmost 10 args
            found = False
            for child, data in children.items():
                if f"arg{i}" not in str(child):
                    continue
                args.append(data[0][0])
                found = True
                break
            if found is False:
                break

        if len(args) == 0:
            # non function components
            return f"{tree_to_code(children['left'])}, {tree_to_code(children['right'])}"
        for i, arg in enumerate(args):
            args[i] = arg.split("_")[-1]
        return ", ".join(args)

    def tree_to_code(tree):
        ((cid, semantics), children) = tree
        m = re.search(r'constant_(\w+)', cid)
        if m and m.group(1) in constants:
            return str(constants[m.group(1)])
        elif not children:
            return cid
        else:
            if set(children.keys()) == set(['left', 'right']) and cid in binary_operators.keys():
                return f"({tree_to_code(children['left'])} {binary_operators[cid]} {tree_to_code(children['right'])})"
            elif set(children.keys()) == set(['argument']) and cid in unary_operators.keys():
                if cid in set(['post-increment', 'post-decrement']):
                    return f"{tree_to_code(children['argument'])}{unary_operators[cid]}"
                else:
                    return f"{unary_operators[cid]}{tree_to_code(children['argument'])}"
            elif set(children.keys()) == set(['condition', 'left', 'right']) and cid == 'guarded-assignment':
                return f"if ({tree_to_code(children['condition'])}) {tree_to_code(children['left'])} = {tree_to_code(children['right'])}"
            else:
                args = get_args_from_children(children)
                return f"{cid}({args})"
    
    return tree_to_code(tree)


def program_to_json(program: Program):
    (tree, constants) = program

    def tree_to_json(tree):
        ((cid, semantics), children) = tree
        json_tree = { "node": cid }
        for k, v in children.items():
            symbols = collect_symbols(semantics, lambda s: ComponentSymbol.parse(s).name == k)
            if any(any_fn(ComponentSymbol.is_rhole, ComponentSymbol.is_lhole)(s) for s in symbols):
                if "children" not in tree:
                    json_tree["children"] = {}
                json_tree["children"][k] = tree_to_json(v)
        return json_tree
    
    json_tree = tree_to_json(tree)
    result = { "tree": json_tree, "constants": constants }
    return result


#TODO: validate
def program_of_json(doc, components) -> Program:

    def tree_of_json(tree, components):
        root = dict(components)[tree["node"]]
        children = {}
        if "children" in tree:
            for hole, subprogram in tree["children"].items():
                children[hole] = tree_of_json(subprogram, components)
        return ((tree["node"], root), children)

    tree = tree_of_json(doc["tree"], components)
    constants = doc["constants"] if "constants" in doc else {}
    return (tree, constants)


# Specification is a map:
#   test id -> (list of paths, test assertion)
# Path is a formula over special trident symbols (angelic, environment, output).
# Test assertion is a formula over trident output symbols.
# The semantics of specification is the satisfiability of
#   for all tests. (V paths) ^ test assertion 
Specification = Dict[str, Tuple[List[Formula], Formula]]


class RuntimeSymbol(Enum):
    ANGELIC  = auto()
    RVALUE   = auto()
    LVALUE   = auto() 
    SELECTOR = auto()
    OUTPUT   = auto()

    def __str__(self):
        return {
            RuntimeSymbol.ANGELIC:  "angelic",
            RuntimeSymbol.RVALUE:   "rvalue",
            RuntimeSymbol.LVALUE:   "lvalue",
            RuntimeSymbol.SELECTOR: "selector",
            RuntimeSymbol.OUTPUT:   "output",
        }[self]

    is_angelic    = lambda s: RuntimeSymbol.check(s) == RuntimeSymbol.ANGELIC
    is_rvalue     = lambda s: RuntimeSymbol.check(s) == RuntimeSymbol.RVALUE
    is_lvalue     = lambda s: RuntimeSymbol.check(s) == RuntimeSymbol.LVALUE
    is_selector   = lambda s: RuntimeSymbol.check(s) == RuntimeSymbol.SELECTOR
    is_output     = lambda s: RuntimeSymbol.check(s) == RuntimeSymbol.OUTPUT
    is_special    = lambda s: RuntimeSymbol.check(s) != None
    is_nonspecial = lambda s: RuntimeSymbol.check(s) == None

    @staticmethod
    def _regex(s):
        return {
            RuntimeSymbol.ANGELIC:  re.compile(r'^choice!angelic!(\w+)!(\d+)!(\d+)$'),
            RuntimeSymbol.RVALUE:   re.compile(r'^choice!rvalue!(\d+)!(\d+)!([\w_\d\[\].]+)$'),
            RuntimeSymbol.LVALUE:   re.compile(r'^choice!lvalue!(\d+)!(\d+)!([\w_\d\[\].]+)$'),
            RuntimeSymbol.SELECTOR: re.compile(r'^choice!lvalue!selector!(\d+)!([\w_\d\[\].]+)$'),
            RuntimeSymbol.OUTPUT:   re.compile(r'^output!(\w+)!(\w+)!(\d+)$')
        }[s]

    @staticmethod
    def check(symbol: Formula) -> Optional['RuntimeSymbol']:
        for s in RuntimeSymbol:
            if RuntimeSymbol._regex(s).search(symbol.symbol_name()):
                return s
        return None

    @staticmethod
    def parse(symbol: Formula) -> SymbolData:
        kind = RuntimeSymbol.check(symbol)
        if kind == None:
            raise ValueError(f"not a valid trident symbol: {symbol}")
        m = re.search(RuntimeSymbol._regex(kind), symbol.symbol_name())
        if kind == RuntimeSymbol.ANGELIC:
            return SymbolData(type=TridentType.parse(m.group(1)), lid=m.group(2), eid=int(m.group(3)))
        elif kind == RuntimeSymbol.RVALUE or kind == RuntimeSymbol.LVALUE:
            return SymbolData(lid=m.group(1), eid=int(m.group(2)), name=m.group(3))
        elif kind == RuntimeSymbol.SELECTOR:
            return SymbolData(lid=m.group(1), name=m.group(2))
        elif kind == RuntimeSymbol.OUTPUT:
            return SymbolData(type=TridentType.parse(m.group(1)), name=m.group(2), eid=int(m.group(3)))
        else:
            logger.error(f"unsupported trident symbol kind {kind}")
            exit(1)

    @staticmethod
    def output(d: SymbolData) -> Formula:
        assert d.type == TridentType.I32  #TODO: support boolean?
        array = Symbol(f"output!{d.type}!{d.name}!{d.eid}", Klee.memory_type)
        return Klee.interpret_memory(array, d.type)

    @staticmethod
    def angelic(d: SymbolData) -> Formula:
        array = Symbol(f"choice!angelic!{d.type}!{d.lid}!{d.eid}", Klee.memory_type)
        return Klee.interpret_memory(array, d.type)

    @staticmethod
    def rvalue(d: SymbolData) -> Formula:
        array = Symbol(f"choice!rvalue!{d.lid}!{d.eid}!{d.name}", Klee.memory_type)
        return Klee.interpret_memory(array, d.type)

    @staticmethod
    def lvalue(d: SymbolData) -> Formula:
        array = Symbol(f"choice!lvalue!{d.lid}!{d.eid}!{d.name}", Klee.memory_type)
        return Klee.interpret_memory(array, d.type)


class Klee:
    memory_type = ArrayType(BV32, BV8)

    @staticmethod
    def interpret_memory(array: Formula, type: TridentType) -> Formula:
        concat = BVConcat(Select(array, BV(3, 32)),
                 BVConcat(Select(array, BV(2, 32)),
                 BVConcat(Select(array, BV(1, 32)), Select(array, BV(0, 32)))))
        if type == TridentType.I32:
            return concat
        elif type == TridentType.BOOL:
            return concat.NotEquals(BV(0, 32))
        raise ValueError(f"unsupported type {type}")

    @staticmethod
    def load(tid: str, assertion: Formula, klee_dir: Path) -> Specification:
        """Load specification for test from KLEE output directory.
        Test-instantiate and path-instantiate all non-special symbols.
        TODO: Ignore paths where only a subset of output variables from assertion is defined.
        TODO: Ignore paths where only a subset of environment variables for a reachable location is defined."""
        smt2s = klee_dir.glob('*.smt2')
        path_formulas = []
        for pathfile in smt2s:
            pid = pathfile.stem
            with pathfile.open() as f:
                pysmt.environment.push_env()
                parser = SmtLibParser()
                script = parser.get_script(f)
                path_formula = script.get_last_formula()
                test_instantiated = Instance.of_formula(path_formula, tid, Instance.TEST, RuntimeSymbol.is_nonspecial)
                path_instantiated = Instance.of_formula(test_instantiated, pid, Instance.PATH, RuntimeSymbol.is_nonspecial)
                pysmt.environment.pop_env()
                normalized = pysmt.environment.get_env().formula_manager.normalize(path_instantiated)
                path_formulas.append(normalized)
        return { tid: (path_formulas, assertion) }


def parse_assertion_symbol(symbol: Formula) -> SymbolData:
    assertion_symbol_regexp = re.compile(r'^([\w_\d]+)!(\d+)$')
    m = re.search(assertion_symbol_regexp, symbol.symbol_name())
    # TODO: support bool type?
    return SymbolData(type=TridentType.I32, name=m.group(1), eid=int(m.group(2)))


def extract_lids(path: Formula) -> Tuple[Set[str], bool]:
    """Extracting lids by checking angelic and lvalue symbols."""
    angelic_symbols = collect_symbols(path, RuntimeSymbol.is_angelic)
    lids = { RuntimeSymbol.parse(s).lid:RuntimeSymbol.parse(s).type for s in angelic_symbols }
    #NB: angelic symbols might be removed by KLEE when they are not used, so check lvalues:
    lvalue_symbols = collect_symbols(path, RuntimeSymbol.is_lvalue)
    lval_candidates = { RuntimeSymbol.parse(s).lid:TridentType.I32 for s in lvalue_symbols
                        if RuntimeSymbol.parse(s).lid not in lids }
    lids.update(lval_candidates)
    return lids, len(lval_candidates) >= 1

def extract_outputs(path: Formula) -> Tuple[Set[str], bool]:
    angelic_symbols = collect_symbols(path, RuntimeSymbol.is_output)
    oids = { RuntimeSymbol.parse(s).lid:RuntimeSymbol.parse(s).type for s in angelic_symbols }
    #NB: angelic symbols might be removed by KLEE when they are not used, so check lvalues:
    output_symbols = collect_symbols(path, RuntimeSymbol.is_output)
    output_candidates = { RuntimeSymbol.parse(s).lid:TridentType.I32 for s in output_symbols
                        if RuntimeSymbol.parse(s).lid not in oids }
    oids.update(output_candidates)
    return oids, len(output_candidates) >= 1


def extract_eids(path: Formula, lid: str) -> Set[str]:
    """Extracting exe ids for given lid by checking angelic and lvalue symbols."""
    symbols = collect_symbols(path, any_fn(RuntimeSymbol.is_angelic, RuntimeSymbol.is_lvalue))
    return set(RuntimeSymbol.parse(s).eid for s in symbols if RuntimeSymbol.parse(s).lid == lid)


# assignment of free variables that satisfy specification
VerificationSuccess = namedtuple('VerificationSuccess', ['constants'])


def get_selector(lvar, specification):
    trigger_set = set()
    for test, pc in specification.items():
        selector_vars = collect_symbols(pc[0][0], RuntimeSymbol.is_selector)
        for selector_var in selector_vars:
            var = selector_var.serialize().split("!")[-1]
            var = var[:-1] if var[-1] == "'" else var
            if lvar == var:
                trigger_set.add(selector_var)

    return trigger_set


def recursive_selector(tree, specification):
    trigger_set = set()
    if tree[0][0] == 'assignment':
        lvar = tree[1]['left'][0][0]
        trigger_set = get_selector(lvar, specification).union(recursive_selector(tree[1]['right'], specification))
    elif type(tree[1]) == dict and len(tree[1]) == 2 and 'left' in tree[1]:
        trigger_set = recursive_selector(tree[1]['left'], specification)
        trigger_set = trigger_set.union(recursive_selector(tree[1]['right'], specification))
    elif tree[0][0] == 'guarded-assignment':
        lvar = tree[1]['left'][0][0]
        trigger_set = get_selector(lvar, specification).union(recursive_selector(tree[1]['right'], specification))
        trigger_set = trigger_set.union(recursive_selector(tree[1]['condition'], specification))
    elif tree[0][0] in ('post-increment', 'post-decrement', 'pre-increment', 'pre-decrement'):
        lvar = tree[1]['argument'][0][0]
        trigger_set = get_selector(lvar, specification)
    elif "func" in tree[0][0]:
        for data in tree[1]:
            trigger_set = trigger_set.union(get_selector(tree[1][data][0][0], specification))
    return trigger_set


def verify(programs: Union[Dict[str, Program], Dict[str, Formula]],
           specification: Specification) -> Optional[VerificationSuccess]:
    """Check if programs satisfy specification
    """
    vc = TRUE()
    program_semantics = { lid:(program if isinstance(program, Formula) else program_to_formula(program))
                          for (lid, program) in programs.items() }
    free_variables = [v for p in program_semantics.values() for v in collect_symbols(p, ComponentSymbol.is_const)]
    selector_vars = collect_symbols(specification[list(specification.keys())[0]][0][0], RuntimeSymbol.is_selector)
    trigger_set = recursive_selector(programs[list(programs.keys())[0]][0], specification)
    condition = TRUE()
    for selector in selector_vars:
        var = Select(selector, BV(0, 32))
        if selector in trigger_set:
            condition = And(condition, Equals(var, BV(1, 8)))
        else:
            condition = And(condition, Equals(var, BV(0, 8)))
    a = 0
    for tid in specification.keys():
        test_vc = FALSE()

        (paths, assertion) = specification[tid]
        #path_id = Symbol("pathvar_test_" + str(tid), BV32)
        for path in paths:
            lids = extract_lids(path)[0]

            assert len(lids) == 1
            lid = list(lids)[0]
            eids = extract_eids(path, lid)

            assert eids == set(range(len(eids)))
            path_vc = path

            program = program_semantics[lid]

            for eid in eids:
                program_substitution = {}
                for program_symbol in collect_symbols(program, lambda x: True):
                    kind = ComponentSymbol.check(program_symbol)
                    data = ComponentSymbol.parse(program_symbol)._replace(lid=lid)._replace(eid=eid)
                    if kind == ComponentSymbol.RRETURN:
                        program_substitution[program_symbol] = RuntimeSymbol.angelic(data)
                    elif kind == ComponentSymbol.RVALUE:
                        program_substitution[program_symbol] = RuntimeSymbol.rvalue(data)
                    elif kind == ComponentSymbol.LVALUE:
                        program_substitution[program_symbol] = RuntimeSymbol.lvalue(data)
                    else:
                        pass #FIXME: do I need to handle it somehow?
                bound_program = program.substitute(program_substitution)
                is_branch = any_fn(ComponentSymbol.is_rbranch, ComponentSymbol.is_lbranch)
                exe_inst_program = Instance.of_formula(bound_program, eid, Instance.EXECUTION, is_branch)
                path_vc = And(path_vc, exe_inst_program)
            test_vc = Or(test_vc, path_vc)
        assertion_substitution = {}
        for assertion_symbol in collect_symbols(assertion, lambda x: True):
            symbol_data = parse_assertion_symbol(assertion_symbol)
            assertion_substitution[assertion_symbol] = RuntimeSymbol.output(symbol_data)
        bound_assertion = assertion.substitute(assertion_substitution)
        test_vc = And(test_vc, bound_assertion, condition)
        is_special_nonconst = any_fn(RuntimeSymbol.is_special, any_fn(all_fn(ComponentSymbol.is_special,
                                                                      complement(ComponentSymbol.is_const)),
                                                                      ComponentSymbol.is_temporary))
        instantiated_test_vc = Instance.of_formula(test_vc, tid, Instance.TEST, is_special_nonconst)
        vc = And(vc, instantiated_test_vc)
        # dump(vc, "vc.smt2")
        a+=1

    dump(vc, "vc.smt2")
    model = get_model(vc)
    if model is None:
        return None
    else:
        return VerificationSuccess({v:model[v].bv_signed_value() for v in free_variables})



def len_exceed(mapping):
    total = 1
    for key in mapping:
        total += len_exceed(mapping[key][1])

    return total




class ReteEnumerator:
    HOLE = "hole"
    def __init__(self, templates: ComponentTree, components, args):
        self.templates = [template[0] for template in templates]
        self.variables = [c for c in components
            if (ComponentSemantics.get_rreturn(c[1]) and c[0] not in COMPONENTS)]
        self.operators = [c for c in components
            if (ComponentSemantics.get_rreturn(c[1]) and c[0] in COMPONENTS)]
        self.constant = [c for c in components
            if (ComponentSemantics.get_rreturn(c[1]) and c[0] == "constant_a")]
        self.ranker = ChainRanker(args.model)
        self.batch = 1
        self.theta = args.theta or 0.263
        self.drop = 100000
        

    def template_to_code(self, template: ComponentTree) -> str:
        return program_to_code((template, {}))

    def has_hole(self, template):
        if self.check_in(template, self.variables + self.constant):
            return False
        if type(template[1]) == dict and len(template[1]) == 2 and 'left' in template[1]:
            lvar = template[1]['left'][0][0]
            rvar = template[1]['right'][0][0]
            if ReteEnumerator.HOLE in (lvar, rvar):
                return True
            if self.has_hole(template[1]['right'][0]):
                return True
        elif template[0][0] == 'guarded-assignment':
            lvar = template[1]['left'][0][0]
            rvar = template[1]['right'][0][0]
            condition = template[1]['condition'][0][0]
            if ReteEnumerator.HOLE in (lvar, rvar, condition):
                return True
            if self.has_hole(template[1]['right'][0]) or self.has_hole(template[1]["condition"][0]):
                return True

        elif template[0][0] in ('post-increment', 'post-decrement', 'pre-increment', 'pre-decrement'):
            lvar = template[1]['argument'][0][0]
            if lvar == ReteEnumerator.HOLE:
                return True
        elif "func" in template[0][0]:
            for data in template[1]:
                if template[1][data][0][0] == ReteEnumerator.HOLE:
                    return True
                if self.has_hole(template[1][data][0]):
                    return True
        return False
    def check_in(self, template, var_list):
        for var in var_list:
            if var[0] == template[0][0]:
                return True
        return False
    def find_and_fill_hole(self, template):
        if template[0][0] == ReteEnumerator.HOLE:
            for i in self.variables + self.constant :
                yield (i, {})

        if type(template[1]) == dict and len(template[1]) == 2 and 'left' in template[1]:
            for l_val in self.iterate_patches(template[1]["left"]):
                yield (template[0], {"right": template[1]["right"], "left": l_val})
            for r_val in self.iterate_patches(template[1]["right"]):
                yield (template[0], {"left": template[1]["left"], "right": r_val})

        elif template[0][0] == 'guarded-assignment':
            for l_val in self.iterate_patches(template[1]["left"]):
                yield (template[0], {"right": template[1]["right"], "left": l_val, "condition": template[1]["condition"]})
            for r_val in self.iterate_patches(template[1]["right"]):
                yield (template[0], {"left": template[1]["left"], "right":r_val, "condition": template[1]["condition"]})
            for c_val in self.iterate_patches(template[1]["condition"]):
                yield (template[0], {"left": template[1]["left"], "condition": c_val, "right": template[1]["right"]})

        elif template[0][0] in ('post-increment', 'post-decrement', 'pre-increment', 'pre-decrement'):
            lvar = template[1]['argument'][0][0]
            for avar in self.iterate_patches(template[1]["argument"]):
                yield (template[0], {"argument": avar})
            
        elif "func" in template[0][0]:
            for arg in template[1]['argument']:
                for i, data in enumerate(self.iterate_patches(arg)):
                    yield (template[0], {f"argument_{i}": data})            

    def syntheize_templates(self, template):
        if self.check_in(template, self.variables + self.constant):
            yield ((ReteEnumerator.HOLE, template[0][1]), template[1])

        if type(template[1]) == dict and len(template[1]) == 2 and 'left' in template[1]:
            yield (template[0], {"right": template[1]["right"], "left": ((ReteEnumerator.HOLE, template[1]["left"][0][1]), {})})
            for r_val in self.syntheize_templates(template[1]["right"]):
                yield (template[0], {"left": template[1]["left"], "right": r_val})

        elif template[0][0] == 'guarded-assignment':
            yield (template[0], {"right": template[1]["right"], "left": ((ReteEnumerator.HOLE, template[1]["left"][0][1]), {}), "condition": template[1]["condition"]})
            for r_val in self.syntheize_templates(template[1]["right"]):
                yield (template[0], {"left": template[1]["left"], "right":r_val, "condition": template[1]["condition"]})
            for c_val in self.syntheize_templates(template[1]["condition"]):
                yield (template[0], {"left": template[1]["left"], "condition": c_val, "right": template[1]["right"]})

        elif template[0][0] in ('post-increment', 'post-decrement', 'pre-increment', 'pre-decrement'):
            lvar = template[1]['argument'][0][0]
            yield (template[0], {"argument": ((ReteEnumerator.HOLE, template[1]["argument"][0][1]), {})})
            
        elif "func" in template[0][0]:
            for data in template[1]:
                for arg_i, data in enumerate(self.iterate_patches(template[1]["arguments"])):
                    yield (template[0], {f"argument_{arg_i}": data})            


    def replace_variables(self, template):
        new_templates = []
        if self.has_hole(template):
            new_templates = self.find_and_fill_hole(template)
        else:
            new_templates = [template]
        return_templates = []
        for template in new_templates:
            return_templates += list(self.syntheize_templates(template))

        return return_templates
            
    def _generate(self, new_patches: List[ComponentTree], edit_dist: Dict[str, int], components: List[Component]):
        new_queue = []
        for template in new_patches:
            template_code = self.template_to_code(template)
            holes = collect_symbols(template[0][1], any_fn(ComponentSymbol.is_lhole, ComponentSymbol.is_rhole))
            if not holes:
                continue
            names = list(set(ComponentSymbol.parse(h).name for h in holes))
            commutative = (template[0][0] in ['addition', 'multiplication',
                                       'equal', 'not-equal',
                                       'logical-and', 'logical-or', 'max', 'bitwise-and'])
            for new_template in self.replace_variables(template):
                code = self.template_to_code(new_template)
                if code in edit_dist:
                    continue
                edit_dist[code] = edit_dist[template_code] + 1
                dist = (math.e**(self.theta * self.ranker.rank(code)/self.batch), new_template)
                new_queue.append((math.e**(self.theta * self.ranker.rank(code)), new_template))

        return new_queue
    
    def iterate_patches(self, template):
        if template[0][0] == ReteEnumerator.HOLE:
            for i in self.variables:
                yield (i, {})

        if type(template[1]) == dict and len(template[1]) == 2 and 'left' in template[1]:
            for l_val in self.iterate_patches(template[1]["left"]):
                yield (template[0], {"right": template[1]["right"], "left": l_val})
            for r_val in self.iterate_patches(template[1]["right"]):
                yield (template[0], {"left": template[1]["left"], "right": r_val})

        elif template[0][0] == 'guarded-assignment':
            for l_val in self.iterate_patches(template[1]["left"]):
                yield (template[0], {"right": template[1]["right"], "left": l_val, "condition": template[1]["condition"]})
            for r_val in self.iterate_patches(template[1]["right"]):
                yield (template[0], {"left": template[1]["left"], "right":r_val, "condition": template[1]["condition"]})
            for c_val in self.iterate_patches(template[1]["condition"]):
                yield (template[0], {"left": template[1]["left"], "condition": c_val, "right": template[1]["right"]})

        elif template[0][0] in ('post-increment', 'post-decrement', 'pre-increment', 'pre-decrement'):
            lvar = template[1]['argument'][0][0]
            for avar in self.iterate_patches(template[1]["argument"]):
                yield (template[0], {"argument": avar})
            
        elif "func" in template[0][0]:
            for data in template[1]:
                pass

    def enumerate_templates(self, components: List[Component],depth: int, typ: TridentType, need_lreturn: bool,
                    need_rreturn: bool, compos_used=None, requires_assignment=False):
        roots = [c for c in components
            if (ComponentSemantics.get_rreturn(c[1]) \
                    and ComponentSymbol.parse(ComponentSemantics.get_rreturn(c[1])).type == typ)]
        edit_dist = {}
        for template in self.templates:
            edit_dist[self.template_to_code(template)] = 0
            yield template
        priority_queue = self._generate(self.templates, edit_dist, components)
        while priority_queue:
            patch = priority_queue.pop(0)
            for new_patch in self.iterate_patches(patch):
                yield new_patch
                priority_queue += self._generate([patch], edit_dist, components)
            priority_queue = sorted(priority_queue, lambda x: x[0])
            priority_queue = priority_queue[:self.drop]
        return 

class MappingDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))

# Enumerating components trees
def enumerate_trees(components: List[Component],
                    depth: int,
                    typ: TridentType,
                    need_lreturn: bool,
                    need_rreturn: bool,
                    compos_used=None, requires_assignment=False):
    roots = [c for c in components
             if (not need_lreturn \
                   or (ComponentSemantics.get_lreturn(c[1]) \
                     and ComponentSymbol.parse(ComponentSemantics.get_lreturn(c[1])).type == typ)) \
                and (not need_rreturn \
                   or (ComponentSemantics.get_rreturn(c[1]) \
                     and ComponentSymbol.parse(ComponentSemantics.get_rreturn(c[1])).type == typ))]

    def enumerate_mappings(names, mapping_params, drop):
        name, *remaining = names
        depth, typ, commutative, need_lreturn, need_rreturn = mapping_params[name]
        subtrees = enumerate_trees(components, depth, typ, need_lreturn, need_rreturn)
        for _ in range(0, drop):
            next(subtrees)
        for (i, substitution) in enumerate(subtrees):
            if remaining:
                next_drop = drop + i if commutative else drop
                for partial_mapping in enumerate_mappings(tuple(remaining), mapping_params, next_drop):
                    if set(partial_mapping.keys()) != set(remaining):
                        yield from ()
                    mapping = {name:substitution}
                    mapping.update(partial_mapping)
                    yield mapping
            else:
                    yield {name:substitution}

    for root in roots:
        if requires_assignment and root[0] not in ('assignment', 'post-decrement', 'post-increment'):
            continue
        holes = collect_symbols(root[1], any_fn(ComponentSymbol.is_lhole, ComponentSymbol.is_rhole))
        if not holes:
            yield (root, {})

        if holes and (depth > 1):
            names = list(set(ComponentSymbol.parse(h).name for h in holes))
            commutative = (root[0] in ['addition', 'multiplication',
                                       'equal', 'not-equal',
                                       'logical-and', 'logical-or', 'max', 'bitwise-and'])
            mapping_params = MappingDict()
            for name in names:
                lhole = ComponentSemantics.get_lhole(root[1], name)
                rhole = ComponentSemantics.get_rhole(root[1], name)
                typ = ComponentSymbol.parse(lhole).type if lhole else ComponentSymbol.parse(rhole).type
                mapping_params[name] = (depth-1, typ, commutative, lhole, rhole)
            for mapping in enumerate_mappings(tuple(names), mapping_params, 0):
                yield (root, mapping)
        


def extract_assigned(tree: ComponentTree) -> List[Component]:
    (root, mapping) = tree
    assigned_names = list(name for name in mapping.keys() if ComponentSemantics.get_lhole(root[1], name))
    assigned_trees = list(mapping[name][0] for name in assigned_names)
    for subtree in mapping.values():
        assigned_trees += extract_assigned(subtree)
    return assigned_trees

def check_equal(tree):
    (root, mapping) = tree
    if root[0] != 'assignment':
        return False
    return mapping['left'] == mapping['right']

def brute_enumeration(components: List[Component],
                    depth: int,
                    typ: TridentType,
                    need_lreturn: bool,
                    need_rreturn: bool,
                    compos_used=None, requires_assignment=False, patch=None):
    
    def recursive_formula_construction(names, used_depth, depth_vals, mapping_params):
        name, *remaining = names
        depth, typ, commutative, need_lreturn, need_rreturn = mapping_params[name]
        for depth_enum in range(1, used_depth):
            for formula in depth_vals[depth_enum]:
                if need_lreturn:
                    if ComponentSemantics.get_lreturn(formula[0][1]) is None or \
                        ComponentSymbol.parse(ComponentSemantics.get_lreturn(formula[0][1])).type != typ:
                        continue
                if need_rreturn:
                    if ComponentSemantics.get_rreturn(formula[0][1]) is None or \
                            ComponentSymbol.parse(ComponentSemantics.get_rreturn(formula[0][1])).type != typ:
                        continue
                if remaining:
                    for mappings in recursive_formula_construction(remaining, used_depth-depth_enum, depth_vals, mapping_params):
                        new_mapping = {name:formula}
                        new_mapping.update(mappings)
                        yield new_mapping
                else:
                    yield  {name:formula}

    
    roots = [c for c in components
             if (not need_lreturn \
                   or (ComponentSemantics.get_lreturn(c[1]) \
                     and ComponentSymbol.parse(ComponentSemantics.get_lreturn(c[1])).type == typ)) \
                and (not need_rreturn \
                   or (ComponentSemantics.get_rreturn(c[1]) \
                     and ComponentSymbol.parse(ComponentSemantics.get_rreturn(c[1])).type == typ))]
    depth_vals = []
    for i in range(depth+1):
        depth_vals.append([])
    used_set = set()
    for used_depth in range(1, depth + 1):
        if used_depth == 1:
            for root in roots:
                if requires_assignment and root[0] not in ('assignment', 'post-decrement', 'post-increment'):
                    continue
                holes = collect_symbols(root[1], any_fn(ComponentSymbol.is_lhole, ComponentSymbol.is_rhole))
                if not holes:
                    code = root[0]
                    if code in used_set:
                        continue
                    used_set.add(code)     
                    depth_vals[1] += [(root, {})]           
                    yield (root, {})
        
            continue
        for root in roots:
            holes = collect_symbols(root[1], any_fn(ComponentSymbol.is_lhole, ComponentSymbol.is_rhole))
            if not holes:
                continue
            names = list(set(ComponentSymbol.parse(h).name for h in holes))
            commutative = (root[0] in ['addition', 'multiplication',
                                       'equal', 'not-equal',
                                       'logical-and', 'logical-or', 'max', 'bitwise-and'])
            mapping_params = MappingDict()
            for name in names:
                lhole = ComponentSemantics.get_lhole(root[1], name)
                rhole = ComponentSemantics.get_rhole(root[1], name)
                typ = ComponentSymbol.parse(lhole).type if lhole else ComponentSymbol.parse(rhole).type
                mapping_params[name] = (depth-1, typ, commutative, lhole, rhole)
            for new_formula in recursive_formula_construction(names, used_depth, depth_vals, mapping_params):
                new_program = (root, new_formula)
                code = program_to_code((new_program, {}))
                if code in used_set:
                    continue
                used_set.add(code)                
                depth_vals[used_depth].append(new_program)      
                yield new_program



def synthesize(components: List[Component],
               depth: int,
               specification: Specification,
               batch_size: int, enumerate_function:object, rank_expr=None) -> Optional[Dict[str, Program]]:
    lids = {}
    requires_assignment = False
    for (tid, (paths, _)) in specification.items():
        for path in paths:
            x, a = extract_lids(path)
            lids.update(x)
            requires_assignment |= a
    logger.info(f"locations extracted from klee paths: {list(lids.keys())}")
    assert len(lids) == 1, ""
    (lid, typ) = list(lids.items())[0]
    batch = []
    total = set()
    rank = 1
    requires_assignment = False
    for tree in enumerate_function(tuple(components), depth, typ, False, True, requires_assignment=requires_assignment):
        assigned = extract_assigned(tree)
        if len(assigned) != len(set(assigned)):
            continue
        if check_equal(tree):
            continue
        code = program_to_code((tree, {}))
        print(code)
        if code in total:
            continue
        if rank%500 == 0:
            print(rank)
        if str(code) == rank_expr:
            print(f"Rank of the expression: {rank}")
            return
        elif rank_expr is not None:
            rank += 1
            continue
        total.add(code)
        result = verify({lid: (tree, {})}, specification)
        if result:
            yield {lid: (batch[0][1], { ComponentSymbol.parse(f).name:v for (f, v) in result.constants.items() })}



def load_specification(spec_files: List[Tuple[Path, Path]]) -> Specification:
    logger.info("loading specification")
    specification: Specification = {}
    smt_parser = SmtLibParser()
    for (assertion_file, klee_dir) in spec_files:
        logger.info(f"loading test assertion {assertion_file}")
        tid = assertion_file.stem

        with assertion_file.open() as f:
            script = smt_parser.get_script(f)
            assertion_formula = script.get_last_formula()
        if tid in specification:
            logger.error(f"directory name {tid} is not unique")
            exit(1)
        logger.info(f"loading klee paths {klee_dir}")
        test_spec = Klee.load(tid, assertion_formula, klee_dir)
        if not test_spec[tid][0]:
            continue
        no_lid = False
        new_paths = []

        for path in test_spec[tid][0]:
            if len(extract_lids(path)[0]) == 0:
                continue
            if len(extract_outputs(path)[0]) == 0:
                continue
            new_paths.append(path)
        if len(new_paths) > 0:
            test_spec[tid] = (new_paths, test_spec[tid][1])
            specification.update(test_spec)
    return specification


def load_programs(files: Dict[str, Path], components: List[Component]) -> Dict[str, Union[Program, Formula]]:
    """Load programs that are represented either semantically through formulas or syntactically through components
    """
    programs = {}
    smt_parser = SmtLibParser()
    for lid, path in files.items():
        if path.suffix == '.smt2':
            with path.open() as smt_file:
                script = smt_parser.get_script(smt_file)
                programs[lid] = script.get_last_formula()
        elif path.suffix == '.json':
            with path.open() as json_file:
                data = json.load(json_file)
                programs[lid] = program_of_json(data, components)
        else:
            logger.error(f"unsupported file type: {path}")
            exit(1)
    return programs


def make_component_from_file(f, cid):
    pysmt.environment.push_env()
    smt_parser = SmtLibParser()
    script = smt_parser.get_script(f)
    semantics = script.get_last_formula()
    is_var_or_const = any_fn(ComponentSymbol.is_rvalue, ComponentSymbol.is_lvalue, ComponentSymbol.is_const)
    # we assume that variables and constants are global, but everything else is local
    instantiated = Instance.of_formula(semantics, cid, Instance.COMPONENT, complement(is_var_or_const))
    pysmt.environment.pop_env()
    normalized = pysmt.environment.get_env().formula_manager.normalize(instantiated)
    return normalized


def load_components(comp_files: List[Path]) -> List[Component]:
    logger.info("loading components")
    components = []
    
    for component_file in comp_files:
        logger.info(f"loading component {component_file}")
        cid = component_file.stem
        COMPONENTS.append(cid)
        if cid in dict(components):
            logger.error(f"file name {cid} is not unique")
            exit(1)
        with component_file.open() as f:
            normalized = make_component_from_file(f, cid)
        components.append((cid, normalized))

    return components


def make_component(symbol:str, lvalue:bool=False) -> Component:
    if lvalue:
        component_str = lvalue_component.format(symbol, symbol, symbol, symbol)
    else:
        component_str = rvalue_component.format(symbol, symbol)
    component = make_component_from_file(io.StringIO(component_str), symbol)
    return (symbol, component)


def get_components(specification: Specification) -> List[Component]:
    components = []
    symbols = set()
    vregex = re.compile(r"([\d\w._\[\]]+)_\d+")
    for test in specification.values():
        for path in test[0]:
            lvalue_symbols = collect_symbols(path, RuntimeSymbol.is_lvalue)
            rvalue_symbols = collect_symbols(path, RuntimeSymbol.is_rvalue)
            rvalue_symbols = [s.symbol_name().split("!")[-1] for s in rvalue_symbols]
            lvalue_symbols = [s.symbol_name().split("!")[-1] for s in lvalue_symbols]
            rvalue_symbols = [vregex.search(v).group(1) if vregex.search(v) else v for v in rvalue_symbols]
            lvalue_symbols = [vregex.search(v).group(1) if vregex.search(v) else v for v in lvalue_symbols]
            for lvalue in lvalue_symbols:
                if lvalue in symbols:
                    continue
                symbols.add(lvalue)
                components.append(make_component(lvalue, lvalue=True))
            for rvalue in rvalue_symbols:
                if rvalue in symbols:
                    continue
                symbols.add(rvalue)
                components.append(make_component(rvalue))
    return components


def main():
    parser = argparse.ArgumentParser('Trident synthesizer')
    parser.add_argument('--tests',
                        nargs='+',
                        type=(lambda a: (a.split(':')[0], a.split(':')[1])),
                        required=True,
                        metavar='FILE:DIR',
                        help='pairs of test assertion and klee paths directory')
    parser.add_argument('--components',
                        nargs='+',
                        metavar='FILE',
                        help='synthesis components')
    parser.add_argument('--verify',
                        nargs='+',
                        type=(lambda a: (a.split(':')[0], a.split(':')[1])),
                        metavar='LID:FILE',
                        help='verify given expressions')
    parser.add_argument('--depth',
                        type=int,
                        default=2,
                        help='depth of synthesized expressions')
    parser.add_argument('--all',
                        action='store_true',
                        help='generate all patches')
    parser.add_argument('--rank',
                    type=str,
                    help='Extracts Rank of the expression, cannot differentiate equivalent expressions')
    parser.add_argument('--templates',
                    nargs='+',
                    metavar='FILE',
                    help='Templates for patch')
    parser.add_argument('--theta',
                    type=int,
                    default=None,
                    help='Constant')
    parser.add_argument('--model',
                    type=str,
                    help='string')

    parser.add_argument('--priority', action='store_true', help='Synthesizes using priority')
    args = parser.parse_args()
    
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    FORMAT = logging.Formatter('%(asctime)s  %(levelname)-7s  %(message)s')
    consoleHandler.setFormatter(FORMAT)
    rootLogger.addHandler(consoleHandler)
    spec_files = [(Path(a[0]), Path(a[1])) for a in args.tests]

    specification = load_specification(spec_files)

    components = get_components(specification)
    if args.components:
        comp_files = [Path(f) for f in args.components]
        components += load_components(comp_files)
    if args.verify:
        program_files = { a[0]: Path(a[1]) for a in args.verify }
        programs = load_programs(program_files, components)
        result = verify(programs, specification)
        if result:
            print("SUCCESS")
        else:
            print("FAIL")
    else:
        if not components:
            logger.error("components are not provided")
            exit(1)
        if args.templates and no_ranker is False:
            template_files = { i: Path(a) for i, a in enumerate(args.templates)}
            templates = load_programs(template_files, components)
            rete_enum = ReteEnumerator(templates.values(), components, args)
            enum = rete_enum.enumerate_templates
        else:
            enum = brute_enumeration

        depth = args.depth
        batch = 100 if args.priority else 1
        result = synthesize(components, depth, specification, batch, enum, rank_expr=args.rank)
        if args.all:
            for i, v in enumerate(result):
                for (lid, prog) in v.items():
                    print(f"#{i} {lid}:\t{program_to_code(prog)}")
                if i == 2:
                    break

        else:
            programs = next(result, None)
            if programs:
                for (lid, prog) in programs.items():
                    print(f"{lid}:\t{program_to_code(prog)}")
            else:
                print("FAIL")


if __name__ == "__main__":
    main()