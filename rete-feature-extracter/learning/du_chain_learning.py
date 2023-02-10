import os
import math
import clang.cindex
import random
from clang.cindex import TokenKind
from clang.cindex import CursorKind
from sys import argv
from keras.backend import var
import pandas as pd
from typing import Tuple, Set
import json
import pickle

import argparse
from time import time
from xgboost import XGBClassifier
from keras import Sequential
from keras import models
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datasets import load_metric
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
from transformers import TrainingArguments, Trainer
from model_trainers import bert_trainer
from copy import deepcopy
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot

clang.cindex.Config.set_library_path("/usr/lib/llvm-12/lib")
metric = load_metric("accuracy")

MUTATION_OPERATORS = [
    "=",    
    "+=",
    "-=",
    "|=",
    "&=",
    "^=",
    "/=",
    "*=",
    "%=",
    "<<=",
    ">>=",
]

POSSIBLE_NODES = {
    "for_loop_init": 0,
    "for_loop_condition": 1,
    "for_loop_increment": 2,
    "loop_condition": 3,
    "l_value": 4,
    "r_value": 5,
    "if": 6,
}
HOLE_TYPES = [
    "for_loop_init", 
    "for_loop_condition", 
    "for_loop_increment", 
    "loop_condition",
    ("l_value", "r_value"),
    "if"
]
FOR_LOOP_MUTATION_NODES = [
    "for_loop_init",
    "for_loop_increment"
]
TYPES = {
    "int": 1,
    "char": 2,
    "bool": 3,
    "long": 4,
    "long long int": 5,
    "long int": 6,
}
OPERATORS = {
    "=": 0, 
    "+": 1, 
    "-": 2, 
    "*": 3, 
    "/": 4, 
    "==": 5, 
    "<<": 6, 
    ">>": 7, 
    ">": 7, 
    "<": 8, 
    "<=": 9, 
    ">=": 10, 
    "++_pre": 11, 
    "--_pre": 12, 
    "++_post": 13, 
    "--_post":14
}
KEYSTUFF = {
    "int": 0,
    "float": 1,
    "long long int": 2,
    "auto": 3,
    "double": 4,
    "struct": 5,
    "const": 6,
    "short": 7,
    "unsigned": 8,
    "break": 9,
    "else": 10,
    "long": 11,
    "switch": 12,
    "continue": 14, 	
    "for": 15,
    "signed": 16,
    "void": 17,
    "case": 18,
    "enum": 19,
    "register": 20,
    "typedef": 21,
    "default": 22,
    "goto":23,
    "sizeof": 24,
    "volatile": 25,
    "char": 26,
    "extern": 27,
    "return": 28,
    "union": 29,
    "do": 30, 	
    "if": 31,
    "static": 32,
    "while": 33,
    "{": 34,
    "}": 35,
    "=": 36, 
    "+": 37, 
    "-": 38, 
    "*": 39, 
    "/": 40, 
    "==": 41, 
    "<<": 42, 
    ">>": 43, 
    ">": 44, 
    "<": 45, 
    "<=": 46, 
    ">=": 47, 
    "++": 48,
    "--": 49
}
FEATURE_LIST = [
    "for_loop_init",
    "for_loop_condition",
    "for_loop_increment",
    "loop_condition",
    "l_value",
    "r_value",
    "if",
    "decl_location",
    "var_type",
    "is_global",
    "current_node_type",
    "program_vars"
]
FEATURE_SIZE = len(FEATURE_LIST)
VOCAB_SIZE = 152
VAR_SIZE = VOCAB_SIZE - len(KEYSTUFF) - 1
SEQ_LEN = 128
MIN_SCOPE = 5
HOLE = 151
gc = 0
glb_indices = []

def lies_in(node_1, node_2):
    return node_1["startColumn"] <= node_2["startColumn"] and node_1["endColumn"] >= node_2["endColumn"]

def construct_feat(new_feat, v_id, row, n_type, var_cnt, var_vals):
    new_feat[v_id * FEATURE_SIZE + POSSIBLE_NODES[n_type]] -= 1
    if n_type == "for_loop_init":
        node_list = [node for node in row if node[0] in ("l_value", "r_value")]
        for node in node_list:
            if lies_in(var_vals, node[1]):
                new_feat[v_id * FEATURE_SIZE + POSSIBLE_NODES[node[0]]] -= 1
    tot = sum(new_feat[v_id * FEATURE_SIZE: v_id * FEATURE_SIZE + len(POSSIBLE_NODES)])
    if tot > 0:
        for i in range(len(POSSIBLE_NODES)):
            new_feat[v_id * FEATURE_SIZE + i] /= tot
    #for operator in var_vals["operators"]:
    #    new_feat[v_id*FEATURE_SIZE + 10 + OPERATORS[operator]] -= 1
    
    new_feat[-2] = POSSIBLE_NODES[n_type]
    new_feat[-1] = var_cnt

def heuristic_1(inputs, outputs):
    def check_accuracy(input, output):
        vector = input
        node_idx = int(vector[-2])
        var_cnt = vector[-1]
        rank_list = []

        for i in range(node_idx, len(vector), FEATURE_SIZE):
            rank_list.append((vector[i], i//FEATURE_SIZE))
        rank_list = sorted(rank_list, reverse=True)
        for i, var in enumerate(rank_list):
            if var[1] == output:
                rank = (1+i)/len(rank_list)
                return rank, (i+1)
                
        assert False
        

    sum = 0
    reciprocal_rank = 0
    for input, output in zip(inputs, outputs):
        acc, rank= check_accuracy(input, output)
        sum += acc
        reciprocal_rank += 1/rank
    return sum/len(outputs), reciprocal_rank/len(outputs)

def encode_neural(token, var_map):
    if token.spelling in KEYSTUFF:
        return KEYSTUFF[token.spelling]
    elif token.spelling not in var_map:
        return len(KEYSTUFF)
    else:
        return len(KEYSTUFF) + 1 + var_map[token.spelling]        

class Parser:
    def __init__(self):
        self.encoded_tokens = []
        self.op_indices = ["=", "+", "-", "*", "/", "==", "<<", ">>", ">", "<", "<=", ">=", "++_pre", "--_pre", "++_post", "--_post", "+="]
        self.sequence_inputs = np.array([])
        self.sequence_outputs = np.array([])
        self.scope_cnt = []
        self.du_chains = {}

    def get_scopes(self, filename, last_line):
        output_path = "output_dir/input_data"
        file_path = os.path.join(output_path, filename)
        index = clang.cindex.Index.create()
        tu = index.parse(file_path)

        output_path = "output_dir/output_data"
        ofile = os.path.join(output_path, filename.split(".")[0]+".json")

        with open(ofile) as f:
            df = json.load(f)

        
        var_map = {}
        id = 0
        scopes = []
        if len(df) == 0:
            return []

        for line_no, row in df.items():
            for nodes in row:
                var_vals = nodes[1]
                if var_vals["varName"] not in var_map:
                    scopes.append((var_vals["startScope"], var_vals["endScope"]))
        
        line_scopes = [0] * (last_line + 1)
        for i in range(1, last_line + 1):
            for scope in scopes:
                if (scope[0] <= i and (i <= scope[1] or scope[1] == -1)) or (scope[0] == 0 and scope[1] == 0):
                    line_scopes[i] += 1
        return line_scopes

    def add_training_data(self, chain, indices, scope_cnt):
        prev_index = 0
        if len(chain) >= SEQ_LEN:
            global gc
            gc += 1
            return
        inputs = []
        outputs = []
        for i, line_idx in enumerate(indices):
            new_chain = deepcopy(chain)
            elements = []
            for t_index in range(prev_index, line_idx):
                if new_chain[t_index] > len(KEYSTUFF) + 1:
                    elements.append(new_chain[t_index])
                    new_chain[t_index] = HOLE
            for index, e in enumerate(elements):
                inputs.append(np.array([0] * (SEQ_LEN - len(new_chain) - 1) + new_chain + [index]))
                outputs.append(np.array(e - len(KEYSTUFF) - 1))
                self.scope_cnt.append(scope_cnt[i])
            prev_index = line_idx
        
        if len(inputs) == 0:
            return    
        
        if len(self.sequence_inputs) == 0:
            self.sequence_inputs = np.vstack(inputs)
            self.sequence_outputs = np.vstack(outputs)
        else:    
            self.sequence_inputs = np.vstack([self.sequence_inputs, np.vstack(inputs)])
            self.sequence_outputs = np.vstack([self.sequence_outputs, np.vstack(outputs)])
    
    def neural_encode(self, chain, var_map):
        new_chain = []
        for token in chain:
            if token in KEYSTUFF:
                new_chain.append(KEYSTUFF[token])
            elif token not in var_map:
                new_chain.append(len(KEYSTUFF))
            else:
                new_chain.append(len(KEYSTUFF) + 1 + var_map[token])
        return new_chain        

    def tokenize(self, file_name, fix_path, is_bert):
        index = clang.cindex.Index.create()
        #file_path = "tests/test2.c"
        tu = index.parse(file_name)
        a = time()
        ofile = os.path.join(fix_path, file_name.split("/")[-1].split(".")[0]+".json")   

        with open(ofile) as f:
            df = json.load(f)
        
        lvals = {}

        var_map = {}
        keys = [int(key) for key in df.keys()]
        id = 0

        try:
            last_line = max([token.location.line for token in tu.get_tokens(extent=tu.cursor.extent)])
        except ValueError:
            return
        if last_line < 50:
            return 
        line_scopes = self.get_scopes(file_name, last_line)
        if len(line_scopes) == 0:
            return

        for key in keys:
            line_data = df[str(key)]
            for data in line_data:
                for lval in data["lvals"]:
                    if lval not in lvals:
                        lvals[lval] = []
                    lvals[lval].append((key, data["startColumn"], data["endColumn"]))
                    if lval not in var_map:
                        var_map[lval] = id
                        id += 1

                for rval in data["rvals"]:
                    if rval not in var_map:
                        var_map[rval] = id
                        id += 1

        filename = tu.cursor.extent.start.file.name
        
        for lval, data in lvals.items():
            for i, re_decl in enumerate(data):
                start = re_decl[0]
                if i + 1 < len(data):
                    end = data[i+1][0]
                else:
                    end = last_line + 1
                
                chain = []
                indices = []
                chain_lines = []
                chain_cnt = 0
                for line in range(start, end):
                    if str(line) not in df:
                        continue
                    for line_data in df[str(line)]:
                        if lval not in line_data["rvals"] or lval not in line_data["lvals"]:
                            continue
                        source_range = ((line, line_data["startColumn"]), (line, line_data["endColumn"] + 1))
                        extent = tu.get_extent(filename, source_range)
                        #chain.append(line_data["node"])
                        for token in tu.get_tokens(extent=extent):
                            chain.append(token.spelling)
                        chain.append(";")
                        try:
                            chain_lines.append(line_scopes[line])
                        except Exception as e:
                            print(line_scopes, line, filename)
                        chain_cnt += len(chain)
                        indices.append(len(chain))
                glb_indices.append(chain_cnt)

                chain = self.neural_encode(chain, var_map)
                self.add_training_data(chain, indices, chain_lines)    
                #chain.append(";")
                #self.du_chains["{}_{}_{}".format(lval, start, filename)] = chain
        
        return

    def add_scope(self, nf, scopes: Tuple[int, int], line_no: int):
        oos_cnt = 0
        scope_vars = []
        for id, scope in scopes.items():
            if not ((scope[0] <= line_no and scope[1] >= line_no) or (scope[1] == -1 and scope[0] <= line_no) or (scope[1] == 0 and scope[0] == 0)):
                oos_cnt += 1
                for i in range(FEATURE_SIZE - 1):
                    nf[id * FEATURE_SIZE + i] = 0
            else:
                scope_vars.append(id)
        return scope_vars

def metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def get_graph_data(lis):
    x_axis_data = sorted(list(set([val[1] for val in lis])))
    y_axis_data = [0]*len(x_axis_data)
    ith_len = [0]*len(x_axis_data)
    for i, line in enumerate(x_axis_data):
        for val, l2 in lis:
            if l2==line:
                y_axis_data[i] += val
                ith_len[i] += 1
    max_i, min_i = max(ith_len), min(ith_len)
    for index, size in enumerate(ith_len):
        y_axis_data[index]/=size
    return np.array(x_axis_data).reshape(-1, 1), np.array(y_axis_data).reshape(-1, 1)




def estimate_neural_prob(test_outputs, predict_outputs, var_cnt):
    assert len(test_outputs) == len(var_cnt)
    rank = 0
    reciprocal_rank = 0 
    r_list = []
    for to, po, cnt in zip(test_outputs, predict_outputs, var_cnt):
        rank_idx = 0
        #rank_list = sorted(rank_list, reverse=True)
        rank_list = np.argsort(po)
        for idx, element in enumerate(rank_list):
            if element == np.argmax(to):
                rank_idx = len(rank_list) - idx 
                break

        r_list.append((rank_idx/cnt, cnt))
        rank += rank_idx/cnt
        reciprocal_rank += 1/rank_idx
    return rank/len(test_outputs), reciprocal_rank/len(test_outputs), r_list

my_parser = argparse.ArgumentParser(description='Args')

my_parser.add_argument('path',
                       metavar='path',
                       type=str,
                       help='the path to files')
my_parser.add_argument('--nn', action='store_true', default=False)
my_parser.add_argument('--samples', action='store', type=int, default=60000)
my_parser.add_argument('--all', action='store_true',  default=False)
my_parser.add_argument('--relearn', action='store_true', default=False)
my_parser.add_argument('--output-path', metavar='path',
                       type=str,
                       help='the path to files')

# Execute the parse_args() method
args = my_parser.parse_args()

i = 0
parser = Parser()

TSIZE = args.samples

for filename in sorted(os.listdir(args.path)):
    a = time()
    try:
        parser.tokenize(args.path + "/" + filename, args.output_path, args.bert)
    except FileNotFoundError as e: 
        raise e
        continue
    i+=1
    if i % 1000 == 0:
        print(i)
    if i>TSIZE:
        break

if args.nn or args.all:

    high = len(parser.sequence_inputs)
    low = int(high * 0.9)
    X_train, X_test, y_train, y_test = parser.sequence_inputs[:low], parser.sequence_inputs[low:high], parser.sequence_outputs[:low], parser.sequence_outputs[low:high]
    y_train = to_categorical(y_train, num_classes=VAR_SIZE)
    y_test = to_categorical(y_test, num_classes=VAR_SIZE)
    a = time()
    if not args.relearn:
        model = models.load_model("chain_nn_node.hdf5")
        model.summary()
    elif args.bert:
        training_args = TrainingArguments(output_dir="train_dir", evaluation_strategy="epoch")

        model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
        trainer = train_bert(model, tokenizer, training_args)
        trainer.save_model("models/")

    else:
        model = Sequential()
        model.add(Embedding(VOCAB_SIZE, 80, input_length=SEQ_LEN))
        model.add(LSTM(60,return_sequences=False))
        model.add(Dense(50, activation='tanh'))
        model.add(Dropout(0.2)) 
        model.add(Dense(40, activation='sigmoid'))
        model.add(Dense(VAR_SIZE, activation='softmax'))# compiling the network
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        mcp_save = ModelCheckpoint('chain_nn_node.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

        model.fit(np.array(X_train), np.array(y_train),batch_size=32,  epochs=100, verbose=1,callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.1)

    outs = model.predict(np.array(X_test))
    prob, mrr, lis_nn = estimate_neural_prob(y_test, outs, parser.scope_cnt[low:high])
    x_axis_nn, y_axis_nn = get_graph_data(lis_nn)
