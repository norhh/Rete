import ast
import sys
from threading import excepthook
from typing import Tuple
from copy import deepcopy
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
import os
from datasets import Dataset
from transformers import AutoTokenizer, FillMaskPipeline, Trainer, DataCollatorWithPadding
from time import time
import argparse
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import transformers as tr
import pathlib
from scipy.special import softmax


def node_equal(node_1, node_2):
    return (
        node_1.lineno == node_2.lineno
        and node_1.end_lineno == node_2.end_lineno
        and node_1.col_offset == node_2.col_offset
        and node_1.end_col_offset == node_2.end_col_offset
    )
def compute_metrics(eval_pred):
    logits, labels = eval_pred
   

    predictions = np.argmax(logits, axis=-1)
    
    acc = np.sum(predictions == labels) / predictions.shape[0]
    return {"accuracy" : acc}

kf = KFold(n_splits=5, shuffle=True, random_state=0)
training_args = tr.TrainingArguments(
    #report_to = 'wandb',
    output_dir='./results_new1', 
    overwrite_output_dir = True,
    num_train_epochs=2,           
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=1000,            
    weight_decay=0.01,
    logging_dir='./logs_new1',
    logging_steps=220,
    evaluation_strategy="epoch"
    ,save_strategy="epoch"
    ,load_best_model_at_end=True
)


def get_target_vars(targets):
    vars = set()
    vars_freq = {}
    if isinstance(targets, ast.Name):
        return {targets}, {targets.id: 1}
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, ast.Name):
                vars.add(target)
                vars_freq[target.id] = vars_freq.get(target.id, 0) + 1
            if isinstance(target, ast.Tuple):
                for var in target.elts:
                    if isinstance(var, ast.Starred):
                        var = var.value
                    if isinstance(var, ast.Subscript):
                        var = var.value
                    while isinstance(var, ast.Attribute):
                        var = var.value
                    if isinstance(var, ast.Tuple) or isinstance(var, ast.List):
                        new_vars, freq = get_target_vars(var)
                        for new_var in new_vars:
                            vars.add(new_var)
                        for element, freq_data in freq.items():
                            vars_freq[element] = vars_freq.get(element, 0) + 1
                        continue
                    if isinstance(var, ast.Name) is False:
                        print(var, var.__dict__)
                    assert isinstance(var, ast.Name)
                    vars.add(var)
                    vars_freq[var.id] = vars_freq.get(var.id, 0) + 1
    return vars, vars_freq


class nodeTransformer(ast.NodeTransformer):
    def __init__(self, node) -> None:
        super().__init__()
        self._rename_node = node

    def visit_Name(self, node):
        if node_equal(node, self._rename_node):
            new_node = deepcopy(node)
            new_node.id = "<mask>"
            return new_node
        else:
            return deepcopy(node)
def tokenise_check(window, code, index):
    return len(tokenizer.tokenize(code[max(index - window, 0): index + window])) < 100

def get_windowed_code(code):
    index = code.find("<mask>")
    low = 10
    high = 10000
    mid = (low + high) // 2
    low = 1000
    while False and high - low > 5:
        if tokenise_check(mid, code, index):
            low = mid
        else:
            high = mid - 1
        mid = (high + low) // 2
    #if tokenise_check(low, code, index):
    return code[max(index - low, 0): index + low]
        
    assert False, "Something wrong happened"
    


def make_dataset(code, nodes, var_decls, tree) -> Tuple[str, str]:
    vars_node = set()
    vars_freq = {}
    for node in nodes:
        vars_dict_1 = {}
        vars_dict_2 = {}
        if isinstance(node, ast.Assign):
            vars, vars_dict_1 = get_target_vars(node.targets)
            vars_node = vars_node.union(vars)
        elif isinstance(node, ast.BinOp):
            vars, vars_dict_1 = get_target_vars(node.right)
            vars_2, vars_dict_2 = get_target_vars(node.left)
            vars_node = vars_node.union(vars_2).union(vars)
        for var, data in vars_dict_1.items():
            vars_freq[var] = vars_freq.get(var, 0) + data
        for var, data in vars_dict_2.items():
            vars_freq[var] = vars_freq.get(var, 0) + data

    test_data = []
    for var in vars_node:
        node_t = nodeTransformer(var)
        new_tree = ast.fix_missing_locations(node_t.visit(deepcopy(tree)))
        new_code = ast.unparse(new_tree)
        assert "<mask>" in new_code
        new_code = get_windowed_code(new_code)
        test_data.append((new_code, var.id, vars_node, vars_freq))
    return test_data


def get_decl(lis):
    var_decl = {}
    for node in lis:
        if not isinstance(node, ast.Assign):
            continue
        vars, _ = get_target_vars(node.targets)
        for var in vars:
            if var.id not in var_decl:
                var_decl[var.id] = set()
            var_decl[var.id].add(node.lineno)
    return var_decl

def predict(test_data):
    fill_mask = pipeline(task="fill-mask", model=model, tokenizer=tokenizer, top_k=100)
    accuracy = 0
    fail = 0
    for test in test_data:
        assert "<mask>" in test[0] 
        a = time()
        try:
            outputs = fill_mask(test[0])
        except IndexError:
            fail += 1
        rank = 0
        detected_rank = (1 + len(test[2]))
        const_flag = 0
        for output in outputs:
            if output["token_str"].replace(" ", "") == test[1]:
                detected_rank = rank
                break
            if output["token_str"].replace(" ", "") not in [var.id for var in test[2]]:
                if const_flag is False:
                    const_flag = True
                    rank += 1
            else:
                rank += 1
        print(detected_rank, test[1], len(test[2]))
        accuracy += detected_rank/(1 + len(test[2])) 
    return accuracy

def train(train_x, train_y):
    counter = 0
    results_lst = []
    for train_idx, val_idx in kf.split(train_x):
        print("Starting fold", counter)

        # split data
        train_x_base = train_x[train_idx].tolist()
        train_y_base = train_y[train_idx].tolist()

        val_texts = train_x[val_idx].tolist()
        val_labels = train_y[val_idx].tolist()

        # do tokenization
        
        train_encodings = tokenizer(train_x_base, truncation=True, padding=True, max_length=512, return_tensors="pt")
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        train_data = {}
        val_data = {}
        for element, label in zip(train_encodings["input_ids"], train_y_base):
            train_data[element] = label
        for element, label in zip(val_encodings["input_ids"], val_labels):
            val_data[element] = label
        #td = Dataset.from_dict(train_data)
        #vd = Dataset.from_dict(val_data)
        # train
        model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
        print(train_data, "TRAIN_DATA")
        trainer = Trainer(
            model=model,                        
            args=training_args,                  
            train_dataset=train_data,         
            eval_dataset=val_data,            
            compute_metrics=compute_metrics
        )
        trainer.train()
    
        # eval
        preds = trainer.predict(val_data)
        
        result_df = pd.DataFrame({
            "text" :val_texts,
            "score" : softmax(preds[0], axis=1)[:,1]
        })
        results_lst.append(result_df)
        
        counter+=1


def predict_heuristic(test_data):
    accuracy = 0
    fail = 0
    for test in test_data:
        order = [val[0] for val in sorted(test[3].items(), key=lambda item: (-item[1]))]
        try:
            rank = order.index(test[1])
        except ValueError:
            rank = 1 + len(order)
        accuracy += rank/(1 + len(order))
    return accuracy

def get_dataset(train_path):
    train_x = []
    train_y = []
    file_cnt = 0
    for root, dirs, files in os.walk(train_path):
        for file in files:
            print(file_cnt)
            code = None
            if ".py" != file[-3:]:
                continue
            print(file, file_cnt)
            try:
                with open(root + "/" + file) as f:
                    code = f.read()
            except Exception as e:
                print(root, dirs, file)
                raise e
                continue
            try:
                parsed_code = ast.parse(code)
                visitor = ParseAttr()
                visitor.visit(parsed_code)
                line_decl_range = get_decl(visitor.data)
                test_data = make_dataset(code, visitor.data, line_decl_range, parsed_code)

                for data in test_data:
                    train_x.append(data[0])
                    train_y.append(data[1])
                file_cnt += 1
                if file_cnt >= 5:
                    break
            except (Exception, AssertionError) as e:
                raise e
        if file_cnt >= 5: break
    print(len(train_x))
    return np.array(train_x), np.array(train_y)

class ParseAttr(ast.NodeVisitor):
    def __init__(self):
        self.data = []

    def visit_BinOp(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.data.append(node)

    def visit_Assign(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.data.append(node)



test_data = []
model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

file_cnt = 0
total_data = 0
accuracy = 0
parser = argparse.ArgumentParser(description='Train/predict berts.')
parser.add_argument('train_path', type=pathlib.Path)
parser.add_argument('--train-bert', action='store_true',
                    default=False,
                    help='train bert')
parser.add_argument('--predict-bert', action='store_true',
                    default=False,
                    help='train bert')
parser.add_argument('--predict-heuristic', action='store_true',
                    default=False,
                    help='train bert')
args = parser.parse_args()
if args.predict_heuristic:
    print("Running predict heuristic")
    for root, dirs, files in os.walk(args.train_path):
        for file in files:
            
            if ".py" != file[-3:]:
                continue
            
            try:
                with open(root + "/" + file) as f:
                    code = f.read()
            except Exception as e:
                print(root, dirs, file)
                raise e
                continue
            try:
                parsed_code = ast.parse(code)
                visitor = ParseAttr()
                visitor.visit(parsed_code)
                a = time()
                line_decl_range = get_decl(visitor.data)
                test_data = make_dataset(code, visitor.data, line_decl_range, parsed_code)
                accuracy += predict_heuristic(test_data)
                total_data += len(test_data)
                file_cnt += 1
                if file_cnt >= 300:
                    break
            except (Exception, AssertionError) as e:
                print(root+file)
                raise e
                print(root + file, file=sys.stderr)
        if file_cnt >= 300: break

    print(accuracy/total_data, total_data, file_cnt)
elif args.predict_bert:
        print("Running predict bert")
        for root, dirs, files in os.walk(args.train_path):
            for file in files:
            
                if ".py" != file[-3:]:
                    continue
                
                try:
                    with open(root + "/" + file) as f:
                        code = f.read()
                except Exception as e:
                    print(root, dirs, file)
                    raise e
                    continue
                try:
                    parsed_code = ast.parse(code)
                    visitor = ParseAttr()
                    visitor.visit(parsed_code)
                    a = time()
                    line_decl_range = get_decl(visitor.data)
                    accuracy += predict(test_data)
                    total_data += len(test_data)
                    file_cnt += 1
                    if file_cnt >= 300:
                        break
                except (Exception, AssertionError) as e:
                    print(root+file)
                    raise e
                    print(root + file, file=sys.stderr)
            if file_cnt >= 300: break

else:
    print("Training bert")
    train_data = {
        [1, 2, 3, 4, 5] : "1"
    }
    trainer = Trainer(
        model=model,                        
        args=training_args,                  
        train_dataset=train_data,         
        eval_dataset=val_data,            
        compute_metrics=compute_metrics
    )
    trainer.train()
    
    # eval
    preds = trainer.predict(val_data)
    
    result_df = pd.DataFrame({
        "text" :val_texts,
        "score" : softmax(preds[0], axis=1)[:,1]
    })
    results_lst.append(result_df)
