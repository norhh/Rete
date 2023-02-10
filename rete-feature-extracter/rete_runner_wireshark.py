from os import listdir
from os.path import join, exists
import subprocess
import sys
from time import time
import multiprocessing as mp
import json

def run_rete(file):
    outfile_name = join("wireshark_chain_data", file.split("/")[-1] + ".json")
    if exists(outfile_name):
        return
    bashCommand = "build/tools/rete {} -get-chain-data -output={}".format(file, outfile_name)
    ret = subprocess.call(bashCommand.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ret != 0:
        print(file, ret)

with open("/rete/wireshark/compile_commands.json") as f:
    data = json.load(f)
    file_list = [element["file"] for element in data]

pool = mp.Pool(8)
for file in file_list:
    if "capsa" not in file:
        continue
    run_rete(file)
exit(0)
pool.map(run_rete, file_list)
pool.close()
