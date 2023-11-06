# t-vFM-Similarity-for-Continual-Learning

Code for [Towards Robust Feature Learning with t-vFM Similarity for Continual Learning](https://arxiv.org/abs/2306.02335).
Our code is based on the implementation of [Co^2L: Contrastive Continual Learning](https://github.com/chaht01/Co2L).

If you find this code useful, please reference in our paper:
```
@inproceedings{DBLP:conf/iclr/GaoK23,
  author       = {Bilan Gao and
                  YoungBin Kim},
  editor       = {Krystal Maughan and
                  Rosanne Liu and
                  Thomas F. Burns},
  title        = {Towards Robust Feature Learning with t-vFM Similarity for Continual
                  Learning},
  booktitle    = {The First Tiny Papers Track at {ICLR} 2023, Tiny Papers @ {ICLR} 2023,
                  Kigali, Rwanda, May 5, 2023},
  publisher    = {OpenReview.net},
  year         = {2023},
  url          = {https://openreview.net/pdf?id=6I5i0Ytnlul},
  timestamp    = {Wed, 19 Jul 2023 17:21:16 +0200},
  biburl       = {https://dblp.org/rec/conf/iclr/GaoK23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

# Run commands are followed implementation of paper Co^2L: Contrastive Continual Learning
## Representation Learning
```
python main.py --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100 --loss tvMFLoss --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine --syncBN
```

## Linear Evaluation
```
python main_linear_buffer.py --learning_rate 1 --target_task 4 --ckpt ./save_random_200/cifar10_models/cifar10_32_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_momentum_1.000_trial_0_500_100_0.2_0.01_1.0_cosine_warm/ --logpt ./save_random_200/logs/cifar10_32_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_momentum_1.000_trial_0_500_100_0.2_0.01_1.0_cosine_warm/
```
