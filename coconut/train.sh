train_script=$1
gpu_id=$2
export FAIRSEQPY=./fairseq-context/
export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo "Start training"
python3 -m source.training.autoscript."${train_script}"
echo "End training"
