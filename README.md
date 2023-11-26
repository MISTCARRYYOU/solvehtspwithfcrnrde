### 1 Introduction

The codes and data for the paper 'Large-scale hybrid task scheduling in cloud-edge collaborative manufacturing systems with FCRN-assisted random differential evolution' published in the journal named 'The International Journal of Advanced Manufacturing Technology'. The paper can be found at: https://doi.org/10.1007/s00170-023-12595-4

The codes contain both the proposed FCRN-RDE and an implementation of several evolution algorithms. Some codes are converted from C++ programs so that parts of them may look a little bit strange. 

The studied problem belongs to a complex scheduling problem, and the proposed method is a type of surrogate-assisted evolutionary algorithm that can accelerate the speed of fitness evaluation in evolutionary algorithms. The codes contain six deep-learning surrogate models, including DNN, RNN, RCNN, TRANSFORMER, FCRN, and CNN. 

### 2 Run

To run the program, just execute:

```python
python main.py
```

The codes are simple but may be really effective in solving some large-scale optimization problems.

### 3 Citation

If you find something that may be helpful for your research, please cite our work as:

@article{wang2023large,
  title={Large-scale hybrid task scheduling in cloud-edge collaborative manufacturing systems with FCRN-assisted random differential evolution},
  author={Wang, Xiaohan and Zhang, Lin and Laili, Yuanjun and Liu, Yongkui and Li, Feng and Chen, Zhen and Zhao, Chun},
  journal={The International Journal of Advanced Manufacturing Technology},
  pages={1--19},
  year={2023},
  publisher={Springer}
}

