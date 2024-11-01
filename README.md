# Intersectional Fairness (ISF)

## Description
Intersectional Fairness (ISF) is a bias detection and mitigation technology for intersectional bias, which combinations of multiple protected attributes cause.  
ISF leverages the existing single-attribute bias mitigation methods to make a machine-learning model fair regarding intersectional bias.  
Approaches applicable to ISF are pre-, in-, and post-processing. For now, ISF supports Adversarial Debiasing, Equalized Odds, Massaging, and Reject Option Classification.

### Supported Python Configurations:

| Item      | Version |
| ------- | -------------- |
| Python  | 3.7 - 3.11|


## Setup
The ISF setup will install resources and patch AIF360.

### Install with `pip`
```bash
pip install git+https://github.com/intersectional-fairness/isf.git
```

### Patch AIF360 for Intersectional Fairness  
Apply a [patch](https://github.com/intersectional-fairness/isf/tree/main/patches) to AIF360 to work with ISF.

The patch contents are as follows.

| file    | method/class | fixes |
| ------- | -------------- | -------------- |
| datasets/structured_dataset.py | validate_dataset | * Changed the generating condition of 'Value Error' condition to support multiple protected attributes |
| algorithms/postprocessing/<br/>reject_option_classification.py | RejectOptionClassification | * Added "F1 difference" to corresponding metric<br/>* Defined "Balanced Accuracy" as default value for accuracy_metric_name |

To apply the patches, run the following command:

```bash
apply-patch-to-aif360-for-isf
```

The above command equivalents to the following command. So you can apply the patches with the following command instead of the above:

```bash
patch {aif360 installed directory path}/datasets/structured_dataset.py structured_dataset.patch
patch {aif360 installed directory path}/algorithms/postprocessing/reject_option_classification.py reject_option_classification.patch
```

## Run the Examples
The `examples` directory contains a diverse collection of jupyter notebooks that use Intersectional Fairness in various ways.  

If you use open data supported by AIF360, you need to download the datasets and place them in their respective directories as described in [aif360/data/README.md in AIF360](https://github.com/Trusted-AI/AIF360/blob/master/aif360/data/README.md).

## Citing Intersectional Fairness
A technical description of Intersectional Fairness is available in this [paper](https://link.springer.com/chapter/10.1007/978-3-030-87687-6_5) (or this [preliminary version](https://arxiv.org/abs/2010.13494)).  
The followings are the bibtex entries for these papers.  

```
@InProceedings{Kobayashi2021-tf,
      title={{One-vs.-One} Mitigation of Intersectional Bias: A General Method for Extending {Fairness-Aware} Binary Classification},
      booktitle={New Trends in Disruptive Technologies, Tech Ethics andArtificial Intelligence},
      author={Kenji Kobayashi and Yuri Nakao},
      publisher={Springer International Publishing},
      pages={43--54},
      year={2021},
      conference={DiTTEt 2021}
}
@misc{kobayashi2020onevsone,
      title={One-vs.-One Mitigation of Intersectional Bias: A General Method to Extend Fairness-Aware Binary Classification},
      author={Kenji Kobayashi and Yuri Nakao},
      year={2020},
      eprint={2010.13494},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
      url = {https://arxiv.org/abs/2010.13494}
}
```

## Support
If you have any questions or problems, please contact [us](MAINTAINERS.md).

## License
Intersectional Fairness and the OSS licenses it uses is [here](LICENSE).

# Fair Oversampling

## Description
The FairOversampling technique is based on the following paper

Author: Ruysuke Sonoda<br>
Title: Fair oversampling technique using heterogeneous clusters<br>
Affiliation: Fujitsu Ltd., 4-1-1 Kamikodanaka, Nakahara-ku, Kawasaki-shi, 211-8588, Kanagawa, Japan

and can be downloaded from<br>
https://www.sciencedirect.com/science/article/abs/pii/S0020025523006448<br>
(see also https://arxiv.org/abs/2305.13875).


The basic idea of the proposed algorithm is the following.

Class imbalance and group (e.g., race, gender, and age) imbalance are acknowledged
as two reasons in data that hinder the trade-off between fairness and utility
of machine learning classifiers. Existing techniques have jointly addressed
issues regarding class imbalance and group imbalance by proposing fair oversampling
techniques. Unlike the common oversampling techniques, which only
address class imbalance, fair oversampling techniques significantly improve the
abovementioned trade-off, as they can also address group imbalance. However,
if the size of the original clusters is too small, these techniques may cause classifier
overfitting. To address this problem, we herein develop a fair oversampling
technique using data from heterogeneous clusters. The proposed technique generates
synthetic data that have class-mix features or group-mix features to make
classifiers robust to overfitting. Moreover, we develop an interpolation method
that can enhance the validity of generated synthetic data by considering the
original cluster distribution and data noise.

For a tutorial on the algorithm and a use case, please, refer to the file

https://github.com/intersectional-fairness/isf/tree/main/src/fair_oversampling/fair_oversampling_tutorial.ipynb

# Fair Downsampling

## Description
The algorithm FairDownsampling can be applied to a wide range of imbalanced datasets.  
The imbalance appears both at the class (majority vs minority) as well as the group level 
(e.g. gender, age, race).  In traditional downsampling methods, such as NearMiss, there is 
usually a trade-off between accuracy and fairness. Downsampling would often slightly increase 
accuracy, but in most cases fairness deteriorates significantly. The proposed downsampling 
algorithm considers both fairness and accuracy, while trying to avoid underfitting, by deleting 
samples away from the class and group boundaries.

For a tutorial on the algorithm and a use case, please, refer to the file

https://github.com/intersectional-fairness/isf/tree/main/src/fair_downsampling/fair_downsampling_tutorial.ipynb
