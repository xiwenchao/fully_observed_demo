# fully_observed_demo

This is a demo code of experiments in our paper:
>KuaiRec: A Fully Observed Dataset for Recommender Systems, [Paper in arXiv](https://arxiv.org/abs/2202.10842).

All the methods applied in the experiments have been clearly demonstrated in previous works and their corresponding code (please refer to https://ear-conv-rec.github.io/manual.html). As a result, we only give a simple example of how to make use of our contributed dataset, **KuaiRec**, to train and evaluate a CRS model. Specifically, the provided code is the Pytorch implementation and evaluation of *Popularity-oriented Recommender* (Section 5.1.5 in the paper) under two settings (MTG & STG) with three different sampling strategies (Uniformly Random, Positivity-oriented and Popularity-oriented). More details are given in the original paper.


Contributors: Shijun Li.

# Environment Requirement
The code has been tested running under Python 3.7.0. The required packages are as follows:
* torch == 1.0.1
* numpy == 1.15.4
* scipy == 1.1.0
* sklearn == 0.19.2

# Data
The original dataset, **KuaiRec**, including the big matrix for traing and small matrix for evaluation, is detailed and provided in https://chongminggao.github.io/KuaiRec/#. 
Here, we have prepossessed the data for building our CRS exactly following the implementation of EAR (https://ear-conv-rec.github.io/manual.html). All necessary data files to run the code are provided in "/data". Specifically, these files contains all the positive samples of each user for training, validation and testing, the embeddings of items, users and attributes derived from FM model, as well as the relations between all items and attributes. 

# File Structure

  ```shell
Demon Code
├── data
│   ├── Multi_target 
│       ├── Popularity_Popularity_multi
│       ├── Popularity_Positivity_multi
│       └── Popularity_Random_multi
│   └── Single_target
│       ├── Popularity_Popularity_single
│       ├── Popularity_Positivity_single
└──       └── Popularity_Random_single
  ```

/data: all prepossessed data files as described above.


/Multi_target/Popularity_Popularity_multi: *Popularity-oriented Recommender* under `MTG` setting with `Popularity-oriented Sampling` strategy.   
/Multi_target/Popularity_Positivity_multi: *Popularity-oriented Recommender* under `MTG` setting with `Positivity-oriented Sampling` strategy.  
/Multi_target/Popularity_Random_multi: *Popularity-oriented Recommender* under `MTG` setting with `Uniformly Random sampling` strategy.


/Single_target/Popularity_Popularity_single: *Popularity-oriented Recommender* under `STG` setting with `Popularity-oriented Sampling` strategy.  
/Single_target/Popularity_Positivity_single: *Popularity-oriented Recommender* under `STG` setting with `Positivity-oriented Sampling` strategy.  
/Single_target/Popularity_Random_single: *Popularity-oriented Recommender* under `STG` setting with `Uniformly Random sampling` strategy.

# Command
* To run the code, please use the command:
```
python run_6.py -mt 15 -playby policy -fmCommand 8 -optim SGD -lr 0.01 -decay 0 -TopKTaxo 3 -gamma 0 -strategy maxsim -startFrom 0 -endAt 10000 -eval 1 -initeval 0 -trick 0 -mini 1 -alwaysupdate 1 -upcount 1 -upreg 0.001 -code 0.301 -purpose train -mod ear -upoptim Ada -uplr 0.01 -alpha 1.0
```
Note that since randomless lies in the model (due to the sampling strategy), it's recommended to repeat the experiments 10 times and calculate the average results.

# Reference

Please kindly cite our paper if you use our codes or dataset.
```
@article{gao2022kuairec,
	title={KuaiRec: A Fully-observed Dataset for Recommender Systems}, 
	author={Chongming Gao and Shijun Li and Wenqiang Lei and Biao Li and Peng Jiang and Jiawei Chen and Xiangnan He and Jiaxin Mao and Tat-Seng Chua},
	year={2022},
	journal={arXiv preprint arXiv:2202.10842},
	primaryClass={cs.IR}
}
```










