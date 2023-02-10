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
import random, os, sys
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf

try:
	from dataloader import TokenList, pad_to_longest
except: pass

class LayerNormalization(Layer):
	def __init__(self, eps=1e-6, **kwargs):
		self.eps = eps
		super(LayerNormalization, self).__init__(**kwargs)
	def build(self, input_shape):
		self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
		self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
		super(LayerNormalization, self).build(input_shape)
	def call(self, x):
		mean = K.mean(x, axis=-1, keepdims=True)
		std = K.std(x, axis=-1, keepdims=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta
	def compute_output_shape(self, input_shape):
		return input_shape

class ScaledDotProductAttention():
	def __init__(self, d_model, attn_dropout=0.1):
		self.temper = np.sqrt(d_model)
		self.dropout = Dropout(attn_dropout)
	def __call__(self, q, k, v, mask):   # mask_k or mask_qk
		attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])  # shape=(batch, q, k)
		if mask is not None:
			mmask = Lambda(lambda x:(-1e+9)*(1.-K.cast(x, 'float32')))(mask)
			attn = Add()([attn, mmask])
		attn = Activation('softmax')(attn)
		attn = self.dropout(attn)
		output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
		return output, attn

class MultiHeadAttention():
	def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
		self.mode = mode
		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v
		self.dropout = dropout
		if mode == 0:
			self.qs_layer = Dense(n_head*d_k, use_bias=False)
			self.ks_layer = Dense(n_head*d_k, use_bias=False)
			self.vs_layer = Dense(n_head*d_v, use_bias=False)
		elif mode == 1:
			self.qs_layers = []
			self.ks_layers = []
			self.vs_layers = []
			for _ in range(n_head):
				self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
				self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
		self.attention = ScaledDotProductAttention(d_model)
		self.layer_norm = LayerNormalization() if use_norm else None
		self.w_o = TimeDistributed(Dense(d_model))

	def __call__(self, q, k, v, mask=None):
		d_k, d_v = self.d_k, self.d_v
		n_head = self.n_head

		if self.mode == 0:
			qs = self.qs_layer(q)  
			ks = self.ks_layer(k)
			vs = self.vs_layer(v)

			def reshape1(x):
				s = tf.shape(x)   
				x = tf.reshape(x, [s[0], s[1], n_head, s[2]//n_head])
				x = tf.transpose(x, [2, 0, 1, 3])  
				x = tf.reshape(x, [-1, s[1], s[2]//n_head])  
				return x
			qs = Lambda(reshape1)(qs)
			ks = Lambda(reshape1)(ks)
			vs = Lambda(reshape1)(vs)

			if mask is not None:
				mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
			head, attn = self.attention(qs, ks, vs, mask=mask)  
				
			def reshape2(x):
				s = tf.shape(x)   
				x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
				x = tf.transpose(x, [1, 2, 0, 3])
				x = tf.reshape(x, [-1, s[1], n_head*d_v])  
				return x
			head = Lambda(reshape2)(head)
		elif self.mode == 1:
			heads = []; attns = []
			for i in range(n_head):
				qs = self.qs_layers[i](q)   
				ks = self.ks_layers[i](k) 
				vs = self.vs_layers[i](v) 
				head, attn = self.attention(qs, ks, vs, mask)
				heads.append(head); attns.append(attn)
			head = Concatenate()(heads) if n_head > 1 else heads[0]
			attn = Concatenate()(attns) if n_head > 1 else attns[0]

		outputs = self.w_o(head)
		outputs = Dropout(self.dropout)(outputs)
		if not self.layer_norm: return outputs, attn
		outputs = Add()([outputs, q])
		return self.layer_norm(outputs), attn

class PositionwiseFeedForward():
	def __init__(self, d_hid, d_inner_hid, dropout=0.1):
		self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
		self.w_2 = Conv1D(d_hid, 1)
		self.layer_norm = LayerNormalization()
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		output = self.w_1(x) 
		output = self.w_2(output)
		output = self.dropout(output)
		output = Add()([output, x])
		return self.layer_norm(output)

class EncoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
	def __call__(self, enc_input, mask=None):
		output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
		output = self.pos_ffn_layer(output)
		return output, slf_attn

class DecoderLayer():
	def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
		self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.enc_att_layer  = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
	def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None, dec_last_state=None):
		if dec_last_state is None: dec_last_state = dec_input
		output, slf_attn = self.self_att_layer(dec_input, dec_last_state, dec_last_state, mask=self_mask)
		output, enc_attn = self.enc_att_layer(output, enc_output, enc_output, mask=enc_mask)
		output = self.pos_ffn_layer(output)
		return output, slf_attn, enc_attn

def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
		if pos != 0 else np.zeros(d_emb) 
			for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
	return pos_enc

def GetPadMask(q, k):
	'''
	shape: [B, Q, K]
	'''
	ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
	mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
	mask = K.batch_dot(ones, mask, axes=[2,1])
	return mask

def GetSubMask(s):
	'''
	shape: [B, Q, K], lower triangle because the i-th row should have i 1s.
	'''
	len_s = tf.shape(s)[1]
	bs = tf.shape(s)[:1]
	mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
	return mask


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
    num_train_epochs=10,           
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

class QANet_Block:
    def __init__(self, dim, n_head, n_conv, kernel_size, dropout=0.1, add_pos=True):
        self.conv = QANet_ConvBlock(dim, n_conv=n_conv, kernel_size=kernel_size, dropout=dropout)
        self.self_att = MultiHeadAttention(n_head=n_head, d_model=dim, 
                                    d_k=dim//n_head, d_v=dim//n_head, 
                                    dropout=dropout, use_norm=False)
        self.feed_forward = PositionwiseFeedForward(dim, dim, dropout=dropout)
        self.norm = LayerNormalization()
        self.add_pos = add_pos

    def __call__(self, x, mask):
        if self.add_pos: 
            x = AddPosEncoding()(x)
            x = self.conv(x)
            z = self.norm(x)
            z, _ = self.self_att(z, z, z, mask)
            x = add_layer([x, z])
            z = self.norm(x)
            z = self.feed_forward(z)
            x = add_layer([x, z])
            return x

class QANet_Encoder:
	def __init__(self, dim=128, n_head=8, n_conv=2, n_block=1, kernel_size=7, dropout=0.1, add_pos=True):
		self.dim = dim
		self.n_block = n_block
		self.conv_first = SeparableConv1D(dim, 1, padding='same')
		self.enc_block = QANet_Block(dim, n_head=n_head, n_conv=n_conv, kernel_size=kernel_size, 
								dropout=dropout, add_pos=add_pos)
	def __call__(self, x, mask):
		if K.int_shape(x)[-1] != self.dim:
			x = self.conv_first(x)
		for i in range(self.n_block):
			x = self.enc_block(x, mask)
		return x



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
    return len(tokenizer.tokenize(code[max(index - window, 0): index + window])) < 80

def get_windowed_code(code):
    index = code.find("<mask>")
    low = 10
    high = 100000
    mid = (low + high) // 2
    while False and high - low > 5:
        if tokenise_check(mid, code, index):
            low = mid
        else:
            high = mid - 1
        mid = (high + low) // 2
    if tokenise_check(low, code, index):
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
    fill_mask = pipeline(task="fill-mask", model=model, tokenizer=tokenizer, top_k=50000)
    accuracy = 0
    fail = 0
    mrr = 0
    for test in test_data:
        assert "<mask>" in test[0] 
        try:
            outputs = fill_mask(test[0])
        except IndexError:
            fail += 1
        rank = 0
        detected_rank = len(outputs)
        const_flag = 0
        rr = 0
        for output in outputs:
            rr += 1
            if output["token_str"].replace(" ", "") == test[1]:
                detected_rank = rr
                break
            if output["token_str"].replace(" ", "") not in [var.id for var in test[2]]:
                if const_flag is False:
                    const_flag = True
                    rank += 1
            else:
                rank += 1
        mrr += 1/(detected_rank + 1)
        print(detected_rank, test[1], len(test[2]))
        accuracy += detected_rank/(1 + len(test[2])) 
    return accuracy, mrr

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
    mrr = 0
    for test in test_data:
        order = [val[0] for val in sorted(test[3].items(), key=lambda item: (-item[1]))]
        try:
            rank = order.index(test[1])
        except ValueError:
            rank = 1 + len(order)
        mrr += 1/(rank+1)
        accuracy += rank/(1 + len(order))
    return accuracy, mrr

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
                    help='predict bert')
parser.add_argument('--predict-heuristic', action='store_true',
                    default=False,
                    help='Use heuristic')
args = parser.parse_args()
mrr = 0
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
                line_decl_range = get_decl(visitor.data)
                test_data = make_dataset(code, visitor.data, line_decl_range, parsed_code)
                a, m = predict_heuristic(test_data)
                accuracy += a
                mrr += m
                total_data += len(test_data)
                file_cnt += 1
                if file_cnt >= 50:
                    break
            except (Exception, AssertionError) as e:
                print(root+file)
                raise e
                print(root + file, file=sys.stderr)
        if file_cnt >= 50: break

    print(accuracy/total_data, total_data, file_cnt, mrr/total_data)
elif args.predict_bert:
        print("Running predict bert")
        mrr=0
        for root, dirs, files in os.walk(args.train_path):
            for file in files:
            
                if ".py" != file[-3:]:
                    continue
                with open(root + "/" + file) as f:
                    code = f.read()
                try:
                    parsed_code = ast.parse(code)
                    visitor = ParseAttr()
                    visitor.visit(parsed_code)
                    line_decl_range = get_decl(visitor.data)
                    test_data = make_dataset(code, visitor.data, line_decl_range, parsed_code)
                    print(len(test_data))
                    a, m = predict(test_data)
                    accuracy += a
                    mrr += m
                    total_data += len(test_data)
                    file_cnt += 1
                    if file_cnt >= 10:
                        break
                except (IndexError) as e:
                    print(root+file)
                    continue
                    print(root + file, file=sys.stderr)
            if file_cnt >= 10: break
        print(mrr/total_data, accuracy/total_data, total_data)

else:
    print("Training bert")
    train_x, train_y = get_dataset(args.train_path)
    train(train_x, train_y)