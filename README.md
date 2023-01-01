# GCNs_on_imbalanced_datasets

This repo provides the code for our SDM'23 paper: [A Linkage-based Doubly Imbalanced Graph Learning Framework for Face Clustering
](https://arxiv.org/abs/2107.02477), by Huafeng Yang, Qijie Shen, Xingjian Chen, Fangyi Zhang and Rong Du.

Due to work changes and differences in the company's internal and external network code systems, if there are any problems, please submit an issue.

##  GCN based face clustering papers collection

1. [Linkage-based Face Clustering via Graph Convolution Network](https://arxiv.org/abs/1903.11306), CVPR 2019 [[Code](https://github.com/Zhongdao/gcn_clustering)]
2. [Learning to Cluster Faces on an Affinity Graph](https://arxiv.org/abs/1904.02749), CVPR 2019 (**Oral**) [[Project Page](http://yanglei.me/project/ltc)]
3. [Learning to Cluster Faces via Confidence and Connectivity Estimation](https://arxiv.org/abs/2004.00445), CVPR 2020 [[Project Page](http://yanglei.me/project/ltc_v2)]
4. [Density-Aware Feature Embedding for Face Clustering](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Density-Aware_Feature_Embedding_for_Face_Clustering_CVPR_2020_paper.pdf), CVPR 2020

## Requirements
* Python >= 3.6
* PyTorch >= 0.4.0
* [faiss](https://github.com/facebookresearch/faiss)

## Datasets

#### 1. Imbalanced MS1M

[Google drive](https://drive.google.com/file/d/1v1s95uS1kP73oR2lSLR9R6lmdVdlnKUj/view?usp=sharing)

#### 2. Imbalanced DeepFashion

[Google drive](https://drive.google.com/file/d/1oIeswFT5TQG8lFmgU4QIPOqLEer3uA62/view?usp=sharing)

## Run

```cd imbalance_gcn ```

#### 1. Imbalanced MS1M

```./run_train_test_dist_sup_random_celeb.sh```

#### 2. Imbalanced DeepFashion

```./run_train_test_dist_sup_random_deepfashion.sh```

## Citation
Please cite the following paper if you use this repository in your reseach.

```
@article{yang2021gcn,
  title={GCN-Based Linkage Prediction for Face Clusteringon Imbalanced Datasets: An Empirical Study},
  author={Yang, Huafeng and Chen, Xingjian and Zhang, Fangyi and Hei, Guangyue and Wang, Yunjie and Du, Rong},
  journal={arXiv preprint arXiv:2107.02477},
  year={2021}
}
```

## Ackownledgement

This repo was developed based [LTC](https://github.com/yl-1993/learn-to-cluster), many thanks to Lei Yang !

