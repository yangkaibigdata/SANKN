# SANKN
The implementation of "Transfer learning based on sparse Gaussian process for regression" in Python. 

Code for the Information Sciences publication. The full paper can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S0020025522004480).
## Contribution

- ANKN algorithm enables flexible modeling of the correlation between the source task and the target task, where three coupled compositional kernel structures can be used to characterize data covariance, and the transfer kernel has more flexible and powerful expression capabilities to extract transfer knowledge.
- TIP algorithm comprehensively considers the correlation between the input data and the output data of the source and the target domains, significantly reducing the computational cost while maintaining the transfer learning performance.


## Setup
This project runs with Python 3.6. Before running the code, you have to install
* [Tensorflow](https:www.tensorflow.org)
* [GPflow-Slim](https://github.com/ssydasheng/GPflow-Slim)

## Dataset
You can find the datasets [here](http://archive.ics.uci.edu/ml/index.php).

## Usage

python exp/regression.py --data energy --split uci_woval --kern nkn
python exp/regression.py --data energy --split uci_woval_pca --kern nkn


Please consider citing if you find this helpful or use this code for your research.

Citation
```
@article{2022Transfer,
  title={Transfer learning based on sparse Gaussian process for regression},
  author={ Yang, K.  and  Lu, J.  and  Wan, W.  and  Zhang, G.  and  Hou, L. },
  journal={Information Sciences: An International Journal},
  number={605-},
  pages={605},
  year={2022},
}
