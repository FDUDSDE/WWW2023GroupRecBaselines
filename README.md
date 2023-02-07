# GroupRecBaselines

## Introduction 

In this repository, we release our implementations of existing group recommendation models.

* Aggregation method: AGREE, GAME
* Hyper-graph Learning: HyperGroup, HCR, HHGR
* Hyper-cude Learning: CubeRec
* Self-supervised Learning: GroupIM, HHGR, CubeRec



Besides, our ConsRec published in WWW'2023 is released [here](https://github.com/FDUDSDE/WWW2023ConsRec).


## Details

Below is the detailed information of our (re)implementation.

|  Name   | Title  | Information |  Comment |
|  ----  |  ----  |  ----  |  ----  |
| [AGREE](http://staff.ustc.edu.cn/~hexn/papers/sigir18-groupRS.pdf) |  Attentive Group Recommendation | SIGIR'2018 | Refactor |
| [GAME](https://dl.acm.org/doi/10.1145/3397271.3401064)  | GAME: Learning Graphical and Attentive Multi-view Embeddings for Occasional Group Recommendation | SIGIR'2020 | Implementation |
| [GroupIM](https://arxiv.org/abs/2006.03736) | GroupIM: A Mutual Information Maximization Framework for Neural Group Recommendation | SIGIR'2020 | Refactor | 
| [HyperGroup](https://arxiv.org/abs/2103.13506) | Hierarchical Hyperedge Embedding-based Representation Learning for Group Recommendation | TOIS'2021 | Implementation
| [HCR](https://ieeexplore.ieee.org/document/9679118/) | Hypergraph Convolutional Network for Group Recommendation | ICDM'2021 | Refactor |
| [HHGR](https://arxiv.org/abs/2109.04200)| Double-Scale Self-supervised Hypergraph Learning for Group Recommendation | CIKM'2021 | Refactor |
| [CubeRec](https://arxiv.org/abs/2204.02592) | Thinking inside The Box: Learning Hypercube Representations for Group Recommendation | SIGIR'2022 | Refactor |


> `Refactor` refers to refactor their official codes to tailor our experimental settings or datasets.


## Cite 
If you make advantages of this repository in your research, please cite the following in your manuscript:
```
@inproceedings{wu2023consrec,
  title={ConsRec: Learning Consensus Behind Interactions for Group Recommendation},
  author={Wu, Xixi and Xiong, Yun and Zhang, Yao and Jiao, Yizhu and Zhang, Jiawei and Zhu, Yangyong and Philip S. Yu},
  booktitle={Proceedings of the ACM Web Conference 2023},
  year={2023},
  organization={ACM}
}
```
