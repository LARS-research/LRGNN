# LRGNN -- Search to Capture Long-range Dependency with Stacking GNNs for Graph Classification
#### This repository is the code for our WWW 2023 paper: [Search to Capture Long-range Dependency with Stacking GNNs for Graph Classification](https://arxiv.org/pdf/2302.08671.pdf)

#### Overview

In this paper, we provide a novel method **LRGNN** to capture the long-range dependencies with stacking GNNs in the graph classification task. We justify that the over-smoothing problem has smaller influence on the graph classification task, and then employ the stacking-based GNNs to extract the long-range dependencies. Two design needs, i.e., sufficient model depth and adaptive skip-connections, are provided when designing the stacking-based GNNs. To meet these two design needs, we unify them into inter-layer connections, and then design these connections with the help of NAS. Extensive experiments demonstrate the rationality and effectiveness of the proposed LRGNN. 

#### Requirements

     torch-geometric==1.7.2
     torch-scatter==2.0.8
     torch==1.8.0+cu111
     numpy==1.18.5
     hyperopt==0.2.7
     python==3.8.3


# Instructions to run the experiment
**Step 1.** Run the search process, given different random seeds.
(The NCI1 dataset is used as an example)

    (B8C1 Full) python train_search.py --data NCI1   --gpu 0 --num_blocks 8 --cell_mode full --num_cells 1 --agg gcn --cos_temp --BN

    (B12C3 Repeat) python train_search.py --data NCI1   --gpu 0 --num_blocks 12 --cell_mode repeat --num_cells 3 --agg gcn --cos_temp --BN

    (B12C3 Diverse) python train_search.py --data NCI1   --gpu 0 --num_blocks 12 --cell_mode diverse --num_cells 3 --agg gcn --cos_temp --BN


The results are saved in the directory `exp_res`, e.g., `exp_res/nci1.txt`.

**Step 2.** Fine tune the searched architectures. You need specify the arch_filename with the resulting filename from Step 1.

    (B8C1 Full) python fine_tune.py --data NCI1 --gpu 0 --num_blocks 8 --num_cells 1 --cell_mode full   --hyper_epoch 20  --arch_filename exp_res/nci1.txt   --cos_lr --BN


#### Evaluation
To reproduce the SOTA performance in Table 2, please use the following code:

    python reproduce.py --data NCI1  --gpu 0


#### Cite
Please kindly cite [our paper](https://arxiv.org/pdf/2302.08671.pdf) if you use this code:  

    @inproceedings{wei2023search,
    title={Search to Capture Long-range Dependency with Stacking GNNs for Graph Classification},
    author={Wei, Lanning and He, Zhiqiang and Zhao, Huan and Yao, Quanming},
    journal={WebConf},
    year={2023}
    }