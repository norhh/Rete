from os import listdir, rename
from os.path import isfile, join, exists
import subprocess
import sys
from time import time
import multiprocessing as mp


def run_rete(file):
    outfile_name = join("dataset/chain_data", file.split(".")[0] + ".json")
    if exists(outfile_name):
        return
    bashCommand = "build/tools/rete {} -get-chain-data -output={}".format(join(sys.argv[1], file), outfile_name)
    ret = subprocess.call(bashCommand.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ret != 0:
        print(file, ret)

pool = mp.Pool(8)
pool.map(run_rete, listdir(sys.argv[1]))
pool.close()
