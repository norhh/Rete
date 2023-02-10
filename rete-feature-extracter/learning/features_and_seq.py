import os
import math
import clang.cindex
import random
from clang.cindex import TokenKind
from clang.cindex import CursorKind
from sys import argv
import pandas as pd
from typing import Tuple, Set
import json
import pickle
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
from transformers import TrainingArguments, Trainer
from model_trainers import bert_trainer

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
import tensorflow as K
from datasets import load_metric

from copy import deepcopy
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot

clang.cindex.Config.set_library_path("/usr/lib/llvm-10/lib")
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
VOCAB_SIZE = 150
VAR_SIZE = VOCAB_SIZE - len(KEYSTUFF) - 1
SEQ_LEN = 128
MIN_SCOPE = 5

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
        self.inputs = []
        self.outputs = []
        self.ooscope_ratio = 0
        self.sequence_inputs = np.array([])
        self.sequence_outputs = np.array([])
        self.seq_var_cnt = []

    def tokenize(self, file_name, fix_path, is_bert):
        index = clang.cindex.Index.create()
        tu = index.parse(file_name)
        a = time()
        ofile = os.path.join(fix_path, file_name.split("/")[-1].split(".")[0]+".json")   

        with open(ofile) as f:
            df = json.load(f)

        if len(df) < 10:
            return
        
        var_map = {}
        id = 0
        scopes = []
        max_line_no = max([int(line_no) for line_no in df])
        for line_no, row in df.items():
            for nodes in row:
                var_vals = nodes[1]
                if var_vals["varName"] not in var_map:
                    scopes.append((var_vals["startScope"], var_vals["endScope"]))
                    var_map[var_vals["varName"]] = id
                    id += 1
        
        if id<5:
            return

        line_scopes = [0] * (max_line_no + 1)
        for i in range(1, max_line_no + 1):
            for scope in scopes:
                if (scope[0] <= i and (i <= scope[1] or scope[1] == -1)) or (scope[0] == 0 and scope[1] == 0):
                    line_scopes[i] += 1
        tokens = [[token, False] for token in tu.get_tokens(extent=tu.cursor.extent) if token.spelling not in (";", "{", "}")]

        for token in tokens:
            line_no = token[0].location.line
            if str(line_no) not in df:
                continue
            for node in df[str(line_no)]: 
                var_vals = node[1]
                if token[0].location.column == var_vals["startColumn"]:
                    token[1] = True
                    break
        
        for token in tokens:
            token.append(token[0].location.line)
            if not is_bert:
                token[0] = encode_neural(token[0], var_map)
        
        seq_inp = []
        seq_out = []
        
        for i, token in enumerate(tokens):
            if token[1]:
                if line_scopes[token[2]] < MIN_SCOPE:
                    continue
                self.seq_var_cnt.append(line_scopes[token[2]])
                assert token[0] -  len(KEYSTUFF) - 1 < VAR_SIZE            
                seq_out.append(np.array(token[0] -  len(KEYSTUFF) - 1))
                diff = max(SEQ_LEN - i, 0)
                seq_inp.append(np.array([0] * diff  + [np.array(token[0]) for token in tokens[max(i-SEQ_LEN, 0):i]]))
  
        if seq_inp == []:
            return        

        if self.sequence_inputs.size != 0:
            self.sequence_inputs = np.concatenate((self.sequence_inputs, np.array(seq_inp)))
        else:
            self.sequence_inputs = np.array(seq_inp)
        if self.sequence_outputs.size != 0:
            self.sequence_outputs = np.concatenate((self.sequence_outputs, np.array(seq_out)))
        else:
            self.sequence_outputs = np.array(seq_out)

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

    def store_cnts(self, filename):
        output_path = "data_dir/output_data"
        ofile = os.path.join(output_path, filename.split(".")[0]+".json")
        with open(ofile) as f:
            df = json.load(f)
        
        if len(df) < 10:
            return

        var_map = {}
        id = 0
        for line_no, row in df.items():
            for nodes in row:
                var_vals = nodes[1]
                if var_vals["varName"] not in var_map:
                    var_map[var_vals["varName"]] = id
                    id += 1
        if id<5:
            return

        var_scopes = {}
        feature_vector = [0] * (FEATURE_SIZE * len(var_map) + 1)
        for line_no, row in df.items():
            for nodes in row:
                
                var_vals = nodes[1]
                node_type = nodes[0]
                v_id = var_map[var_vals["varName"]]
                
                var_scopes[v_id] = (var_vals["startScope"], var_vals["endScope"]) 
                feature_vector[v_id * FEATURE_SIZE + POSSIBLE_NODES[node_type]] += 1

                feature_vector[v_id * FEATURE_SIZE + 7] = var_vals["declLocation"]
                feature_vector[v_id * FEATURE_SIZE + 8] = TYPES.get(var_vals["varType"], 7)
                feature_vector[v_id * FEATURE_SIZE + 9] = var_vals["isGlobal"]

                #for operator in var_vals["operators"]:
                #    feature_vector[v_id*FEATURE_SIZE + 10 + OPERATORS[operator]] += 1
                
        for line_no, row in df.items():
            new_feat = deepcopy(feature_vector)
            for i in range(7, FEATURE_SIZE * len(var_map), FEATURE_SIZE):
                if new_feat[i] == -1:
                    continue
                new_feat[i] = abs(new_feat[i] - int(line_no))
            
            scope_vars = self.add_scope(new_feat, var_scopes, int(line_no))
            if len(scope_vars) < MIN_SCOPE:
                continue
            for hole in HOLE_TYPES:
                nf = deepcopy(new_feat)
                for node_type, var_vals in row:
                    
                    if node_type != hole or node_type not in hole:
                        continue                        
                    
                    v_id = var_map[var_vals["varName"]]
                    if v_id not in scope_vars:
                        continue

                    construct_feat(nf, v_id , row, node_type, len(scope_vars), var_vals)
                    
                    self.inputs.append(np.array(nf))
                    self.outputs.append(v_id)
        
        self.ooscope_ratio /= len(df)

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

def metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def estimate_prob(test_inputs, test_outputs, predict_outputs):
    rank = 0
    r_list = []
    reciprocal_rank = 0
    for ti, to, po in zip(test_inputs, test_outputs, predict_outputs):
        rank_idx = int(ti[-1])
        max_val = 0
        #rank_list = sorted(rank_list, reverse=True)
        rank_list = np.argsort(po[:int(ti[-1])])
        for idx, element in enumerate(rank_list):
            if element == to:
                rank_idx = len(rank_list)  - idx 
                break
        
        r_list.append((rank_idx/ti[-1], ti[-1]))
        rank += rank_idx/ti[-1]
        reciprocal_rank += 1/rank_idx
    return rank/len(test_inputs), reciprocal_rank/len(test_inputs),  r_list         

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
my_parser.add_argument('--xgb', action='store_true', default=False)
my_parser.add_argument('--nn', action='store_true', default=False)
my_parser.add_argument('--rf', action='store_true', default=False)
my_parser.add_argument('--dt', action='store_true', default=False)
my_parser.add_argument('--samples', action='store', type=int, default=60000)
my_parser.add_argument('--all', action='store_true',  default=False)
my_parser.add_argument('--relearn', action='store_true', default=False)
my_parser.add_argument('--relearn-xgb', action='store_true', default=False)
my_parser.add_argument('--bert', action='store_true', default=False)

# Execute the parse_args() method
args = my_parser.parse_args()

i = 0
parser = Parser()

TSIZE = args.samples

for filename in sorted(os.listdir(args.path)):
    try:
        #parser.tokenize(filename)
        parser.store_cnts(filename)
    except FileNotFoundError:
        print(filename)
        continue
    i+=1
    if i % 1000 == 0:
        print(i)
    if i>TSIZE:
        break



if args.all:
    acc, mrr = heuristic_1(parser.inputs, parser.outputs)
    print("Heuristic", acc, mrr)

max_len = max([len(i) for i in parser.inputs])
for i in range(len(parser.inputs)):
    parser.inputs[i] = np.concatenate((parser.inputs[i][:-2], np.array((max_len - len(parser.inputs[i])) * [0]), parser.inputs[i][-2:]))

high = len(parser.inputs)
low = int(high * 0.9)

X_train, X_test, y_train, y_test = parser.inputs[:low], parser.inputs[low:high], parser.outputs[:low], parser.outputs[low:high]

if args.xgb or args.all:
    filename = 'XGB_model.sav'
    if not args.relearn and not args.relearn_xgb:
        clf = pickle.load(open(filename, 'rb'))
    else:
        clf = XGBClassifier()
        clf = clf.fit(np.array(X_train), np.array(y_train))
        pickle.dump(clf, open(filename, 'wb'))
    
    importance = clf.feature_importances_
    # summarize feature importance
    priority_xgb = [0] * len(FEATURE_LIST)
    for i,v in enumerate(importance):
        priority_xgb[i%len(FEATURE_LIST)] += v

    predict_outputs = clf.predict_proba(np.array(X_test))
    prob, mrr, xgb_lis = estimate_prob(X_test, y_test, predict_outputs)
    x_axis_xg, y_axis_xg = get_graph_data(xgb_lis)
    cnt = {}
    for ele in xgb_lis:
        if ele[1] in cnt:
            cnt[ele[1]] += 1
        else:
            cnt[ele[1]] = 1


    print("XGB", prob, mrr)
    del clf


if args.dt or args.all:
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(np.array(X_train), np.array(y_train))
    priority_dtree = [0] * len(FEATURE_LIST)
    for i,v in enumerate(clf.feature_importances_):
        priority_dtree[i%len(FEATURE_LIST)] += v

    predict_outputs = clf.predict_proba(X_test)
    prob, mrr, lis_dt = estimate_prob(X_test, y_test, predict_outputs)
    cnt = {}
    for ele in lis_dt:
        if ele[1] in cnt:
            cnt[ele[1]] += 1
        else:
            cnt[ele[1]] = 1
    for i, j in enumerate(cnt):
        print(i, j)

    print("DTree", prob, mrr)
    del clf

if args.rf or args.all:
    filename = 'RF_model.sav'
    if False and not args.relearn:
        clf_ensemble = pickle.load(open(filename, 'rb'))
    else:
        clf_ensemble = RandomForestClassifier()
        clf_ensemble = clf_ensemble.fit(X_train, y_train)
        pickle.dump(clf_ensemble, open(filename, 'wb'))
    priority_rf = [0] * len(FEATURE_LIST)


    for i,v in enumerate(clf_ensemble.feature_importances_):
        priority_rf[i%len(FEATURE_LIST)] += v


    predict_outputs = clf_ensemble.predict_proba(X_test)
    prob, mrr,  lis_rf = estimate_prob(X_test, y_test, predict_outputs)


    del clf_ensemble
    x_axis_rf, y_axis_rf = get_graph_data(lis_rf)
    print("Random Forest", prob, mrr)

if False:
    range_ = np.arange(len(priority_rf))

    plt.bar(range_ + 0.00, priority_xgb, color = 'b', width = 0.25)
    plt.bar(range_ + 0.33, priority_rf, color = 'g', width = 0.5)
    plt.bar(range_ + 0.66, priority_dtree, color = 'r', width = 0.75)

    plt.ylabel('Feature Importance')
    plt.title('Featue Importance')
    plt.xticks(0.25 + np.arange(len(FEATURE_LIST)) , FEATURE_LIST, fontsize=7)
    
    plt.legend(labels=['Random Forest', 'Decision Tree', 'xgb'])
    plt.show()


if args.nn or args.all:

    high = len(parser.sequence_inputs)
    low = int(high * 0.9)
    X_train, X_test, y_train, y_test = parser.sequence_inputs[:low], parser.sequence_inputs[low:high], parser.sequence_outputs[:low], parser.sequence_outputs[low:high]
    y_train = to_categorical(y_train, num_classes=VAR_SIZE)
    y_test = to_categorical(y_test, num_classes=VAR_SIZE)

    if not args.relearn:
        model = models.load_model("chain_nn.hdf5")
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
        mcp_save = ModelCheckpoint('neural_network_model_medium.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

        model.fit(np.array(X_train), np.array(y_train),batch_size=32,  epochs=100, verbose=1,callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.1)

    outs = model.predict(np.array(X_test))

    prob, mrr, lis_nn = estimate_neural_prob(y_test, outs, parser.seq_var_cnt[low:high])
    print("NN", prob, mrr)

    x_axis_nn, y_axis_nn = get_graph_data(lis_nn)

if args.plot:
    plt.xlabel("In scope vars")
    plt.ylabel("search efficiency")
    if args.all or args.rf:
        plt.plot(x_axis_rf, y_axis_rf, "b", label="Random Forest")
        plt.legend()
    if args.xgb or args.all:
        plt.plot(x_axis_xg, y_axis_xg, "g", label="XGB")
        plt.legend()
    if args.nn or args.bert or args.all:
        plt.plot(x_axis_nn, y_axis_nn, "r", label="NN")
        plt.legend()

    plt.show()
