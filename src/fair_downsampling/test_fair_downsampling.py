# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2024 Fujitsu Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Note: Packages in ./requirements.txt have to be installed before running the test.
from xgboost import XGBClassifier
from imblearn.under_sampling import NearMiss #controlled under-sampling algorithm
from ucimlrepo import fetch_ucirepo #for downloading datasets

import aif360
from aif360 import metrics

import fairdownsampling as fd
from pathlib import Path

#import adult dataset
here = Path(__file__).parent
adult_dataset = pd.read_csv(Path(here, 'adult.csv'))

#convert Adult dataset into AIF360 dataset format
aif360_adult = aif360.datasets.BinaryLabelDataset(
    df=adult_dataset,
    label_names=["income"],
    favorable_label=1.0,
    unfavorable_label=0.0,
    protected_attribute_names=["sex"],
    unprivileged_protected_attributes=[np.array([0.0])],
    privileged_protected_attributes=[np.array([1.0])])

#expected metrics after downsampling for the following seven classification algorithms
#LogisticRegression, RandomForestClassifier, KNeighborsClassifier,
#XGBClassifier, GaussianNB, GradientBoostingClassifier, DecisionTreeClassifier

expected_balanced_accuracy = [
    0.7328019315288707,
    0.853015788229943,
    0.744302657939512,
    0.8746933470744023,
    0.7641738513461461,
    0.8643610977196046,
    0.8138945661332111
]

expected_statistical_parity_difference = [
    0.013005464480874307,
    -0.02016890213611522,
    0.05790797317436658,
    -0.01549801291604569,
    0.10174615002483856,
    -0.006828738201689055,
    0.013818306010928982
]

expected_equal_opportunity_difference = [
    0.10859171254708788,
    -0.09961825809784175,
    0.028378667808952596,
    -0.11706909150466782,
    0.17390910637118423,
    -0.0804602950625527,
    -0.08295405868494321
]

expected_average_odds_difference = [
    0.025677501309104678,
    -0.022509735791396868,
    0.058735432269084335,
    -0.01963873669539655,
    0.11264140776830597,
    -0.00844020700007532,
    0.009228158104790017
]

#downsample
model_adult = fd.FairDownSampler(sampling_strategy=0.8)
downsampled_adult = model_adult.fit_resample(aif360_adult)

X_resampled = ((downsampled_adult.convert_to_dataframe())[0]).drop(columns=['income'])
y_resampled = ((downsampled_adult.convert_to_dataframe())[0])['income']

#calculated accuracy and fairness metrics after downsampling for various classifiers
models = [LogisticRegression(max_iter=10000),
          RandomForestClassifier(),
          KNeighborsClassifier(),
          XGBClassifier(),
          GaussianNB(),
          GradientBoostingClassifier(),
          DecisionTreeClassifier()
         ]

calculated_balanced_accuracy = []
calculated_statistical_parity_difference = []
calculated_equal_opportunity_difference = []
calculated_average_odds_difference = []
    
for model in models:

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.30, random_state=40)
    
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test.values)

    dataset_before = X_test.copy()
    dataset_before["Labels"] = y_test.copy()
    aif360_before = aif360.datasets.BinaryLabelDataset(
        df=dataset_before,
        label_names=["Labels"],
        protected_attribute_names=["sex"])

    dataset_after = X_test.copy()
    dataset_after["Labels"] = y_predicted.copy()
    aif360_after = aif360.datasets.BinaryLabelDataset(
        df=dataset_after,
        label_names=["Labels"],
        protected_attribute_names=["sex"])

    classification_metric = aif360.metrics.ClassificationMetric(
        dataset=aif360_before,
        classified_dataset=aif360_after,
        unprivileged_groups=[{"sex": 0}],
        privileged_groups=[{"sex": 1}])

    calculated_balanced_accuracy.append(balanced_accuracy_score(y_test, y_predicted))
    calculated_statistical_parity_difference.append(-classification_metric.statistical_parity_difference())
    calculated_equal_opportunity_difference.append(-classification_metric.equal_opportunity_difference())
    calculated_average_odds_difference.append(-classification_metric.average_odds_difference())

def test_fair_downsampling():
    
    calculated_1 = np.array(calculated_balanced_accuracy)
    calculated_2 = np.array(calculated_statistical_parity_difference)
    calculated_3 = np.array(calculated_equal_opportunity_difference)
    calculated_4 = np.array(calculated_average_odds_difference)

    expected_1 = np.array(expected_balanced_accuracy)
    expected_2 = np.array(expected_statistical_parity_difference)
    expected_3 = np.array(expected_equal_opportunity_difference)
    expected_4 = np.array(expected_average_odds_difference)

    tolerance = np.full(7, 1e-3)
    
    assert (np.abs(calculated_1 - expected_1) < tolerance).any(), f'Error, got {calculated_1}, expected {expected_1}'
    assert (np.abs(calculated_2 - expected_2) < tolerance).any(), f'Error, got {calculated_2}, expected {expected_2}'
    assert (np.abs(calculated_3 - expected_3) < tolerance).any(), f'Error, got {calculated_3}, expected {expected_3}'
    assert (np.abs(calculated_4 - expected_4) < tolerance).any(), f'Error, got {calculated_4}, expected {expected_4}'
