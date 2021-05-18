# Social Collaborative Metric Learning

Implementation of Social Collaborative Metric Learning (SCML).

# Environment
Tested on python 3.6 and TensorFlow 1.14.0.

# Example
Run the code with this example:
```bash
python SCML.py --manifold Euclidean --cuda 0 -d 10 --lr 0.001 --dataset ciao --rating_margin 0.1 --social_margin 0.1 --Lambda 0.1
```

# Citation
If you find our work helpful, please cite the following paper:

*Sixiao Zhang, Hongxu Chen, Xiao Ming, Lizhen Cui, Hongzhi Yin, and Guandong Xu. 2021. Where are we in embedding spaces? A ComprehensiveAnalysis on Network Embedding Approaches for Recommender Systems. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21)*

# Acknowledgement
This code is developed based on Collaborative Metric Learning (https://github.com/changun/CollMetric)