# CoCoNut-Artifact
This is DIRECTLY taken and modified from the reproduction package of CoCoNut artifact.
```
@article{lutellier2020coconut,
  title={CoCoNuT: Combining Context-Aware Neural Translation Models using Ensemble for Program Repair},
  author={Lutellier, Thibaud and Pham, Viet Hung and Pang, Lawrence and Li, Yitong and Wei, Moshi and Tan, Lin},
  booktitle={Proceedings of the 28th ACM SIGSOFT International Symposium on Software Testing and Analysis},  
  year={2020}
}
```

For example:
```
python $fairseq_dir/train.py --use-context --fp16 --save-dir $trg_dir/model --arch fconv_context --distributed-world-size 1 --encoder-embed-dim 250 --decoder-embed-dim 250 --decoder-out-embed-dim 250 --encoder-layers '[(256,3)] * 7' --decoder-layers '[(256,3)] * 7' --dropout 0.2 --clip-norm 0.1 --lr 0.25 --min-lr 1e-4 --momentum 0.99 --max-epoch 1 --batch-size 32 $trg_dir/bin
```
