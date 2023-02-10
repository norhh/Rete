import ast
import argparse
import math
import os
import pathlib
from enum import Enum
import pickle
import numpy as np
from typing import List
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
N_COMPS = 1000

class ProphetLabels(Enum):
    VAR = 0
    CONST0 = 1
    CONSTN0 = 2 
    COND = 3 
    IF = 4
    PRT = 5 
    LOOP = 6
    EE = 7
    NE = 8
    EL = 11 
    ER = 12
    IS = 11
    ISNOT = 12
    IN = 13
    NOTIN = 14
    LT_R = 15
    PRINT = 16
    LT_L = 17
    GT_R = 18
    GT_L = 19
    GTE_L = 20
    GTE_R = 21
    LTE_L = 22
    LTE_R = 23
    ADD_L = 24
    ADD_R = 25
    MULT_L = 26
    MULT_R = 27
    SUB_L = 28
    SUB_R = 29
    MATMULT_L = 30
    MATMULT_R = 31
    DIV_L = 32
    DIV_R = 33
    POW_L = 34
    POW_R = 35
    MOD_L = 36
    MOD_R = 37
    LSHIFT_L = 38
    LSHIFT_R = 39
    RSHIFT_L = 40
    RSHIFT_R = 41
    BITOR_L = 42
    BITOR_R = 43
    BITXOR_L = 44
    BITXOR_R = 45
    BITAND_L = 46
    BITAND_R = 47
    FLOORDIV_L = 48
    FLOORDIV_R = 49
    # These are added as python supports `x = a or b`
    AND_L = 50
    AND_R = 51
    OR_L = 52
    OR_R = 53
    ATTR_L = 54
    ATTR_R = 55

def node_hash(node):
    return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)

class Atoms(ast.NodeVisitor):
    def __init__(self):
        self.atoms = dict()

    def visit_Assign(self, node):
        ast.NodeVisitor.generic_visit(self, node)
    
    def visit_Constant(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        if node.lineno not in self.atoms:
            self.atoms[node.lineno] = set()
        if type(node.value) != str:
            self.atoms[node.lineno].add(node.value)
        elif node.value != "print":
            self.atoms[node.lineno].add("STRING_CONSTANT: "+node.value)
        
    def visit_Name(self, node):
        """
        Original Prophet strictly restricts this to variables in C which is different from python's defn.
        Since functions in python are variables.
        :param node: [description]
        :type node: [type]
        """
        ast.NodeVisitor.generic_visit(self, node)
        if node.lineno not in self.atoms:
            self.atoms[node.lineno] = set()
        if node.id != "print":
            self.atoms[node.lineno].add(node.id)

def satisfies(atom, node):
    if isinstance(node.left, ast.Name) and node.left.id == atom:
        return "L"
    if isinstance(node.right, ast.Name) and node.right.id == atom:
        return "R"
    if isinstance(node.right, ast.Constant) and node.right.value == atom:
        return "R"
    return False

def get_all_children(node):
    class ChildrenFinder(ast.NodeVisitor):
        def __init__(self):
            self.children = set()

        def visit_Name(self, node):
            ast.NodeVisitor.generic_visit(self, node)
            self.children.add(node)

        def visit_Constant(self, node):
            ast.NodeVisitor.generic_visit(self, node)
            self.children.add(node)

    finder = ChildrenFinder()
    finder.visit(node)
    return finder.children

def covers(n1, n2):
    if n1[0][0] != n2[0][0]:
        return False
    if n1[0][1] <= n2[0][1] and n1[0][3] >= n2[0][3]:
        return True
    return False 

def prune_children(nodes: List):
    eliminate = {}
    for i in range(len(nodes)):
        if eliminate.get(i, False) is True:
            continue 
        for j in range(i + 1, len(nodes)):
            if covers(nodes[i], nodes[j]):
                eliminate[j] = True
        eliminate[i] = False
    new_nodes = []
    for i in range(len(nodes)):
        if not eliminate[i]:
            new_nodes.append(nodes[i])
    return new_nodes

class ProphetFeatures(ast.NodeVisitor):
    def __init__(self, atoms):
        self.data = dict()
        self.atoms = atoms

    def visit_BinOp(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        for atom in self.atoms[node.lineno]:
            self.data[(node_hash(node), atom)] = set()
            if not satisfies(atom, node):
                continue
            self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(node.left), atom)])
            if node.op == ast.Eq:
                self.data[(node_hash(node), atom)].add(ProphetLabels.COND)
                self.data[(node_hash(node), atom)].add(ProphetLabels.EE)
            elif node.op == ast.NotEq:
                self.data[(node_hash(node), atom)].add(ProphetLabels.COND)
                self.data[(node_hash(node), atom)].add(ProphetLabels.NE)
            elif node.op == ast.Is:
                self.data[(node_hash(node), atom)].add(ProphetLabels.COND)
                self.data[(node_hash(node), atom)].add(ProphetLabels.IS)
            elif node.op == ast.IsNot:
                self.data[(node_hash(node), atom)].add(ProphetLabels.COND)
                self.data[(node_hash(node), atom)].add(ProphetLabels.ISNOT)
            elif node.op == ast.In:
                self.data[(node_hash(node), atom)].add(ProphetLabels.COND)
                self.data[(node_hash(node), atom)].add(ProphetLabels.IN)
            elif node.op == ast.NotIn:
                self.data[(node_hash(node), atom)].add(ProphetLabels.COND)
                self.data[(node_hash(node), atom)].add(ProphetLabels.NOTIN)
            
            elif node.op == ast.Lt:
                self.data[(node_hash(node), atom)].add(ProphetLabels.COND)
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.LT_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.LT_R)


            elif node.op == ast.LtE:
                self.data[(node_hash(node), atom)].add(ProphetLabels.COND)
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.LTE_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.LTE_R)

            elif node.op == ast.GtE:
                self.data[(node_hash(node), atom)].add(ProphetLabels.COND)
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.GTE_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.GTE_R)

            elif node.op == ast.Gt:
                self.data[(node_hash(node), atom)].add(ProphetLabels.COND)
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.GT_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.GT_R)
            elif node.op == ast.Add:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.ADD_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.ADD_R)
            elif node.op == ast.Sub:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.SUB_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.SUB_R)
            elif node.op == ast.Mult:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.MULT_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.MULT_R)

            elif node.op == ast.MatMult:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.MATMULT_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.MATMULT_R)
            
            elif node.op == ast.Div:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.DIV_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.DIV_R)
            elif node.op == ast.Mod:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.MOD_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.MOD_R)
            elif node.op == ast.Pow:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.POW_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.POW_R)

            elif node.op == ast.LShift:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.LSHIFT_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.LSHIFT_R)
            
            elif node.op == ast.RShift:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.RSHIFT_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.RSHIFT_R)
            elif node.op == ast.BitOr:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.BITOR_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.BITOR_R)
            elif node.op == ast.BitXor:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.BITXOR_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.BITXOR_R)
            elif node.op == ast.BitAnd:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.BITAND_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.BITAND_R)
            
            elif node.op == ast.FloorDiv:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.FLOORDIV_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.FLOORDIV_R)

            elif node.op == ast.And:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.AND_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.AND_R)
            elif node.op == ast.FloorDiv:
                if satisfies(atom, node) == "L":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.OR_L)
                elif satisfies(atom, node) == "R":
                    self.data[(node_hash(node), atom)].add(ProphetLabels.OR_R)


    def visit_Assign(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        
        for atom in self.atoms[node.lineno]:
            l = None
            r = None
            for target in node.targets:
                if "elts" in target.__dict__:
                    for element in target.elts:
                        if element.id == atom:
                            l = element
                            break
            for right_node in get_all_children(node.value):
                if isinstance(right_node, ast.Name) and right_node.id == atom:
                    r = right_node
                    break
            self.data[(node_hash(node), atom)] = set()
            if l:
                self.data[(node_hash(node), atom)] = set([ProphetLabels.EL])
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(l), atom)])
            if r:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(set([ProphetLabels.ER]))
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(r), atom)])
    def visit_Attribute(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        for atom in self.atoms[node.lineno]:
            l, r = None, None
            for l_node in get_all_children(node.value):
                if isinstance(l_node, ast.Name) and l_node.id == atom:
                    l = l_node
                    break
            self.data[(node_hash(node), atom)] = set()
            if node.attr == atom:
                r = atom
            if l:
                self.data[(node_hash(node), atom)].add(ProphetLabels.ATTR_L)
            if r:
                self.data[(node_hash(node), atom)].add(ProphetLabels.ATTR_R)
    def visit_AugAssign(self, node):
            
        ast.NodeVisitor.generic_visit(self, node)
        for atom in self.atoms[node.lineno]:
            l = None
            r = None
            if node.target.id == atom:
                l = node.target
            for right_node in get_all_children(node.value):
                if isinstance(right_node, ast.Name) and right_node.id == atom:
                    r = right_node
                    break
            self.data[(node_hash(node), atom)] = set()
            if l:
                self.data[(node_hash(node), atom)] = set([ProphetLabels.EL])
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(l), atom)])
            if r:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(set([ProphetLabels.ER]))
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(r), atom)])
            
    def visit_AnnAssign(self, node):
            
        ast.NodeVisitor.generic_visit(self, node)
        
        for atom in self.atoms[node.lineno]:
            l = None
            r = None
            if node.target.id == atom:
                l = node.target
            for right_node in get_all_children(node.value):
                if isinstance(right_node, ast.Name) and right_node.id == atom:
                    r = right_node
                    break
            self.data[(node_hash(node), atom)] = set()
            if l:
                self.data[(node_hash(node), atom)] = set([ProphetLabels.EL])
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(l), atom)])
            if r:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(set([ProphetLabels.ER]))
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(r), atom)])


    def visit_Constant(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        for atom in self.atoms[node.lineno]:
            if atom == node.value:
                if type(node.value) == int and node.value == 0:
                    self.data[(node_hash(node), atom)] = set([ProphetLabels.CONST0])
                elif type(node.value) == int:
                    self.data[(node_hash(node), atom)] = set([ProphetLabels.CONSTN0])
                else:
                    self.data[(node_hash(node), atom)] = set()
            else:
                self.data[(node_hash(node), atom)] = set()



        
    def visit_If(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        for atom in self.atoms[node.lineno]:
            self.data[(node_hash(node), atom)] = self.data.get((node_hash(node.test), atom), set())
            for stmt in node.body:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data.get((node_hash(stmt), atom), set()))
            for else_node in node.orelse:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(else_node), atom)])

            self.data[(node_hash(node), atom)].add(ProphetLabels.IF)

    def visit_IfExp(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        for atom in self.atoms[node.lineno]:
            self.data[(node_hash(node), atom)] = self.data.get((node_hash(node.test), atom), set())
            if type(node.body) == list:
                for stmt in node.body:
                    self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data.get((node_hash(stmt), atom), set()))
            else:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data.get((node_hash(node.body), atom), set()))
            if type(node.orelse) == list:
                for else_node in node.orelse:
                    self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(else_node), atom)])
            else:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(node.orelse), atom)])


            self.data[(node_hash(node), atom)].add(ProphetLabels.IF)

    def visit_While(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        for atom in self.atoms[node.lineno]:
            self.data[(node_hash(node), atom)] = self.data.get((node.test.lineno, node_hash(node.test), atom), set())
            for stmt in node.body:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data.get((node_hash(stmt), atom), set()))

            for else_node in node.orelse:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(else_node), atom)])

            self.data[(node_hash(node), atom)].add(ProphetLabels.LOOP)

    def visit_For(self, node):
        # TODO: Support adding features for Call nodes in For
        ast.NodeVisitor.generic_visit(self, node)
        for atom in self.atoms[node.lineno]:
            self.data[(node_hash(node), atom)] = self.data.get((node_hash(node.target), atom), set())
            for stmt in node.body:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data.get((node_hash(stmt), atom), set()))
            for else_node in node.orelse:
                self.data[(node_hash(node), atom)] = self.data[(node_hash(node), atom)].union(self.data[(node_hash(else_node), atom)])

            self.data[(node_hash(node), atom)].add(ProphetLabels.LOOP)

        
    def visit_Name(self, node):
        """
        Original Prophet strictly restricts this to variables in C which is different from python's defn.
        Since functions in python are variables.
        :param node: [description]
        :type node: [type]
        """
        ast.NodeVisitor.generic_visit(self, node)
        for atom in self.atoms[node.lineno]:
            if atom == node.id:
                self.data[(node_hash(node), atom)] = set([ProphetLabels.VAR])
            else:
                self.data[(node_hash(node), atom)] = set()

def calc_g(theta, feature, mod_features):
    
    num_g = math.e**(np.dot(feature, theta)[0][0])
    den_g = 0
    for feat in mod_features:
        den_g += math.e**(np.dot(feat, theta)[0][0])
    return num_g/den_g

def get_rank(theta, data, correct_loc):
    all_data = []
    for i in range(5):
        for patch in data[i]:
            all_data.append(data[i])
    correct_val = calc_g(theta, data[correct_loc[0]][correct_loc[1]], all_data)
    rank = 0
    for i in range(5):
        for patch in data[i]:
            new_val = calc_g(theta, patch, all_data)
            if correct_val > new_val or math.isclose(correct_val, new_val):
                rank += 1
    return rank

def update_learning(data, ttheta, n0):
    """
    Equivalent C code of `delta` from original paper
            for (size_t i = 0; i < T.size(); i++) {
            std::vector<double> a;
            const TrainingCase &c = T[i];
            a.resize(c.cases.size());
            double sumExp = 0;
            for (size_t j = 0; j < c.cases.size(); j++) {
                a[j] = Theta.dotProduct(c.cases[j]);
                sumExp += exp(a[j]);
            }
            for (size_t j = 0; j < c.cases.size(); j++) {
                double p = exp(a[j])/sumExp;
                const FeatureVector &vec = c.cases[j];
                for (size_t k = 0; k < vec.size(); k++) {
                    unsigned int idx = vec[k];
                    delta[idx] -= p;
                }
            }
            resT -= log(sumExp);
        }
        resT /= T.size();
        double adjustedResT = resT;
        for (size_t i = 0; i < delta.size(); i++) {
            delta[i] = delta[i] / T.size() - 2 * lambda * Theta[i] - lambdal1 * get_sign(Theta[i]) ;
            adjustedResT -= lambda * Theta[i] * Theta[i] - lambdal1 * fabs(Theta[i]);
        }
    """
    l = 0.01
    regularisation = -2 * l * ttheta - l * (np.ones(ttheta.shape) * np.sign(ttheta))
    log_val = 0
    delta = 0
    for key in data.keys()[:n0]:
        mod_f = []
        for i in range(5):
            for feat in data[key][i]:
                mod_f.append(feat)
        for i in range(5):
            for feat in data[key][i]:
                val = calc_g(ttheta, feat, mod_f) 
                log_val += math.log(val) 
                delta -= feat
    
    return delta + regularisation

def refine_data(data, correct_loc, keys):
    correct_dict = {}
    train_y = []
    train_x = []
    for key in keys:
        train_y.append(1)
        train_x.append(data[key][correct_loc[key][0]][correct_loc[key][1]])
        correct_feat = data[key][correct_loc[key][0]][correct_loc[key][1]]
        for i in range(5):
            for feat in data[key][i]:
                if feat == correct_feat:
                    continue
                train_x.append(feat)
                train_y.append(1)
    return train_x, train_y

parser = argparse.ArgumentParser(description='Train/predict berts.')
subparsers = parser.add_subparsers(dest="command", help="Commands")

feature_parser = subparsers.add_parser(
        "extract-features",
        help="Check functions which are completely safe using symbolic execution",
        formatter_class=RawTextHelpFormatter,
    )
feature_parser.add_argument('file_path', type=pathlib.Path)
feature_parser.add_argument('--output-file',
                    type=pathlib.Path,
                    help='Output pickle path')
vector_parser = subparsers.add_parser(
        "feature-vector",
        help="Check functions which are completely safe using symbolic execution",
        formatter_class=RawTextHelpFormatter,
    )

vector_parser.add_argument('--buggy',
                    type=pathlib.Path,
                    help='buggy pickle features')
vector_parser.add_argument('--correct',
                    type=pathlib.Path,
                    help='correct pickle features')
vector_parser.add_argument('--mod-kind',
                    type=int,
                    help='An int in the list ordered by index: {InsertControl, InsertGuard, ReplaceCond, ReplaceStmt, InsertStmt}')

vector_parser.add_argument('--line-no',
                    type=int,
                    help='Line number')
vector_parser.add_argument('--output-file',
                    type=int,
                    help='Line number')

# TODO: Move prophet training into this file!
train_parser = subparsers.add_parser(
        "train",
        help="Check functions which are completely safe using symbolic execution",
        formatter_class=RawTextHelpFormatter,
    )
train_parser.add_argument('--data-path',
                    type=pathlib.Path,
                    help='path of the dataset')

train_parser.add_argument('--rf',
                    type=bool,
                    default=False,
                    help='Use random forest over that ancient prophet learning algorithm')

args = parser.parse_args()

if args.command == "extract-features":
    with open(args.file_path) as f:
        code = f.read()
        parsed_code = ast.parse(code)
        visitor = Atoms()
        visitor.visit(parsed_code)
        visitor = ProphetFeatures(visitor.atoms)
        parsed_code = ast.parse(code)
        visitor.visit(parsed_code)
        with open(args.output_file, "wb") as write_file:
            pickle.dump(visitor.data, write_file)
elif args.command == "feature-vector":  

    with open(args.buggy, "rb") as fp:
        buggy_data = pickle.load(fp)

    with open(args.correct, "rb") as fp:
        correct_data = pickle.load(fp)



    lineno = args.line_no
    prev_range = [lineno-3, lineno-2, lineno-1]

    next_range = [lineno+1, lineno+2, lineno+3]

    prev_nodes = []
    next_nodes = []
    current_nodes = []

    for key in buggy_data.keys():
        if key[0][0] in prev_range:
            prev_nodes.append(key)
        elif key[0][0] in next_range:
            next_nodes.append(key)
        elif key[0][0] == lineno:
            current_nodes.append(key)
    # TODO: Handle loops better
    prev_nodes = [node for node in prev_nodes if node[0][0] == node[0][2]]
    current_nodes = [node for node in current_nodes if node[0][0] == node[0][2]]
    next_nodes = [node for node in next_nodes if node[0][0] == node[0][2]]

    prev_nodes = prune_children(prev_nodes)
    current_nodes = prune_children(current_nodes)
    next_nodes = prune_children(next_nodes)
    fv = [0]*2048

    prev_nodes_fixed = []
    next_nodes_fixed = []
    current_nodes_fixed = []

    for key in correct_data.keys():
        if key[0][0] in prev_range:
            prev_nodes_fixed.append(key)
        elif key[0][0] in next_range:
            next_nodes_fixed.append(key)
        elif key[0][0] == lineno:
            current_nodes_fixed.append(key)
    # TODO: Handle loops better
    prev_nodes_fixed = [node for node in prev_nodes_fixed if node[0][0] == node[0][2]]
    current_nodes_fixed = [node for node in current_nodes_fixed if node[0][0] == node[0][2]]
    next_nodes_fixed = [node for node in next_nodes_fixed if node[0][0] == node[0][2]]

    prev_nodes_fixed = prune_children(prev_nodes_fixed)
    current_nodes_fixed = prune_children(current_nodes_fixed)
    next_nodes_fixed = prune_children(next_nodes_fixed)
    fv = [0]*12000

    feat_dict = dict()
    feat_index = 1
    for n in current_nodes_fixed:
        for s in prev_nodes:
            for ac in buggy_data[s]:
                for ac_ in correct_data[n]:
                    fv[60*ac.value + ac_.value] = 1
        for s in current_nodes:
            for ac in buggy_data[s]:
                for ac_ in correct_data[n]:
                    fv[3600+ 60*ac.value + ac_.value] = 1

        for s in current_nodes:
            for ac in buggy_data[s]:
                for ac_ in correct_data[n]:
                    fv[7200 + 60*ac.value + ac_.value] = 1

    fv[10800] = args.mod_kind

    with open(args.output_file) as f:
        pickle.dump(fv, f)

# Moved this to the same file
elif args.command == "train":
    data = {}
    correct_loc = {}
    for root, dirs, files in os.walk(args.train_path):
        for file in files:
            with open(file, "rb") as f:
                mod_index = int(file.split("-")[1])
                file_index = file.split("-")[0]
                correct = "c" in file.split("-")[2]
                if file_index not in data:
                    data[file_index] = [[], [], [], [], []]
                if correct:
                    correct_loc[file_index] = (mod_index, len(data[file_index][mod_index]))
                data[file_index][mod_index].append(pickle.load(f))
    n = len(data.keys())
    keys = data.keys()

    if args.rf:
        train_x, train_y = refine_data(data, correct_loc, keys)
        pca = PCA(n_components=N_COMPS)
        train_x = pca.fit_transform(train_x)
        clf_ensemble = RandomForestClassifier()
        clf_ensemble = clf_ensemble.fit(train_x, train_y)
        pickle.dump(clf_ensemble, open("prophet_weight.pcl", 'wb'))
    else:
        n0 = n*0.85
        theta = np.zeros((12000, 1))
        alpha = 1
        gamma = 1
        cnt = 0
        ttheta = np.zeros((12000, 1))
        # Unlike the original paper, we assume perfect localisation through and through!
        while cnt < 200:
            tgamma = 0
            ttheta += alpha*update_learning(data, ttheta, n0)
            for i in range(n0+1, n):
                tot = 0
                for mods in data[keys[i]]:
                    tot += len(mods)
                rank = get_rank(theta, data[keys[i]], correct_loc[keys[i]])
                tgamma += rank/(tot*(n - n0))

            if tgamma < gamma:
                theta = ttheta
                gamma = tgamma
                print("Best Updated")
                cnt = 0
            else:
                cnt += 1
                # The implementation uses this! The paper gives a different value.
                if alpha > 1:
                    print(f"Drop Alpha from {alpha} to {0.1 * alpha}")
                    alpha *= 0.1
        
        with open("prophet_weight.pcl", "wb") as weight_file:
            weight_file.write(theta)


    



