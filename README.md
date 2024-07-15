# GUGEN
# GUGEN: Global User Graph Enhanced Network For Next POI Recommendation

This is a pytorch implementation of 《GUGEN: Global User Graph Enhanced Network For Next POI Recommendation》(IEEE TMC 2024, under review). 


## Abstract

Learning the next Point-of-Interest (POI) is a highly context-dependent human movement behavior prediction task, which has gained increasing attention with the consideration of massive spatial-temporal trajectories data or check-in data. The spatial dependency, temporal dependency, sequential dependency and social network dependency are widely considered pivotal to predict the users’ next location in the near future. However, most existing models fail to consider the influence of other users’ movement patterns and the correlation with the POIs the user has visited. Therefore, we propose a Global User Graph Enhanced Network (GUGEN) for the next POI recommendation from a global and a user perspectives. Firstly, a trajectory learning network is designed to model the users’ short-term preference. Second, a geographical learning module is designed to
model the global and user context information. From the global perspective, two graphs are designed to represent the global POI features and the geographical relationships of all POIs. From the user perspective, a user graph is constructed to describe each users’ historical POI information. We evaluated the proposed model on three real-world datasets. The experimental evaluations demonstrate that the proposed GUGEN method outperforms the state-of-the-art approaches for the next POI recommendation.

## Installation

- Install Pytorch 1.8.1 (Note that the results reported in the paper are obtained by running the code on this Pytorch version. As raised by the issue, using higher version of Pytorch may seem to have a performance decrease on optic cup segmentation.)
- Clone this repo

```
git clone https://github.com/CQRhinoZ/GUGEN
```

## Project Structure

- train_model.py: The training code of our model
- utils.py: Dataset code
- Model.py: main file for Model
- requirements.txt: List the pip dependencies

## Dependency

After installing the dependency:

    pip install -r requirements.txt

## Train

- Download datasets from [here](https://drive.google.com/drive/folders/1o72mNxUgSJX43KcQ2Tg_YE3N3oJEZX5T?usp=drive_link).
- Run `train_model.py`.


## Citation

```
GUGEN: Global User Graph Enhanced Network For Next POI Recommendation, IEEE TMC 2024 (under review)
```

Feel free to contact us:

Xu ZHANG, Ph.D, Professor

Chongqing University of Posts and Telecommunications

Email: zhangx@cqupt.edu.cn

Website: https://faculty.cqupt.edu.cn/zhangx/zh_CN/index.htm
