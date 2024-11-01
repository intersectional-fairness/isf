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
from imblearn.under_sampling import NearMiss #controlled under-sampling
import aif360

class FairDownSampler:
    """
    The algorithm FairDownsampling can be applied to a wide range of imbalanced datasets.  The imbalance appears both at the class (majority vs minority) as well as the group level (e.g. gender, age, race).  In traditional downsampling methods, such as NearMiss, there is usually a trade-off between accuracy and fairness. Downsampling would often slightly increase accuracy, but in most cases fairness deteriorates significantly. The proposed downsampling algorithm considers both fairness and accuracy, while trying to avoid underfitting, by deleting samples away from the class and group boundaries.  Limitations: One protected attribute.

    Parameters
    ----------
    dataset : StructuredDataset
        Input aif360 dataset
    sampling_strategy
        amount of desired downsampling

    Returns
    ----------
    dataset: StructuredDataset
        Downsampled aif360 dataset
    """
  
    def __init__(self, sampling_strategy):
        """
        it constructs a new FairDownSampler object
        """
        self.sampling_strategy = sampling_strategy
    
    def fit_resample(self, aif360_dataset):
        """
        It downsamples a given aif360 dataset

        Parameters
        ----------
        dataset : StructuredDataset
            Input aif360 dataset
            
        Returns
        ----------
        dataset: StructuredDataset
            Downsampled aif360 dataset
        """
        
        self.aif360_dataset = aif360_dataset
        aif360_before = self.aif360_dataset

        #read parametres from input dataset
        label_names = aif360_before.label_names
        favorable_label = aif360_before.favorable_label
        unfavorable_label = aif360_before.unfavorable_label
        protected_attribute_names = aif360_before.protected_attribute_names
        unprivileged_protected_attributes = aif360_before.unprivileged_protected_attributes
        privileged_protected_attributes = aif360_before.privileged_protected_attributes

        #split dataset according to classes and groups
        X = (aif360_before.convert_to_dataframe())[0]
        group_1 = X[(X[protected_attribute_names[0]]==unprivileged_protected_attributes[0][0]) 
                    & (X[label_names[0]]==unfavorable_label)]
        group_2 = X[(X[protected_attribute_names[0]]==unprivileged_protected_attributes[0][0]) 
                    & (X[label_names[0]]==favorable_label)]
        group_3 = X[(X[protected_attribute_names[0]]==privileged_protected_attributes[0][0]) 
                    & (X[label_names[0]]==unfavorable_label)]
        group_4 = X[(X[protected_attribute_names[0]]==privileged_protected_attributes[0][0]) 
                    & (X[label_names[0]]==favorable_label)]

        #we want to downsample group_1 and group_3
        #we further randomly split group_1 and group_3 into three equal parts

        group_1a = group_1.sample(int(len(group_1)/3), random_state=0)
        group_1b = group_1.drop(group_1a.index).sample(int(len(group_1)/3), random_state=0)
        group_1c = group_1.drop(group_1a.index).drop(group_1b.index)

        group_3a = group_3.sample(int(len(group_3)/3), random_state=0)
        group_3b = group_3.drop(group_3a.index).sample(int(len(group_3)/3), random_state=0)
        group_3c = group_3.drop(group_3a.index).drop(group_3b.index)

        #the different groups to downsample are
        Group_1 = pd.concat([group_1a, group_2])
        Group_2 = pd.concat([group_1b, group_4])
        Group_3 = pd.concat([group_3a, group_2])
        Group_4 = pd.concat([group_3b, group_4])
        #groups 1c and 3c have the same labels, we will temporarily change them
        group_3c[label_names[0]] = favorable_label
        Group_5 = pd.concat([group_1c, group_3c])

        #prepare for downsampling by isolating X and y
        X_Group_1 = Group_1.drop(columns=label_names)
        X_Group_2 = Group_2.drop(columns=label_names)
        X_Group_3 = Group_3.drop(columns=label_names)
        X_Group_4 = Group_4.drop(columns=label_names)
        X_Group_5 = Group_5.drop(columns=label_names)
        
        y_Group_1 = Group_1[label_names[0]]
        y_Group_2 = Group_2[label_names[0]]
        y_Group_3 = Group_3[label_names[0]]
        y_Group_4 = Group_4[label_names[0]]
        y_Group_5 = Group_5[label_names[0]]

        #calculate sizes of downsampled datasets
        n1 = len(group_2)/self.sampling_strategy
        n2 = len(group_4)/self.sampling_strategy
        
        my_dict_1 = {favorable_label: len(group_2), unfavorable_label: int(n1 / 3)}
        my_dict_2 = {favorable_label: len(group_4), unfavorable_label: int(n1 / 3)}
        my_dict_3 = {favorable_label: len(group_2), unfavorable_label: int(n2 / 3)}
        my_dict_4 = {favorable_label: len(group_4), unfavorable_label: int(n2 / 3)}
        my_dict_5 = {favorable_label: int(n2 / 3), unfavorable_label: int(n1 / 3)}
        
        #downsampling
        rus_Group_1 = NearMiss(version=2, sampling_strategy=my_dict_1)
        rus_Group_2 = NearMiss(version=2, sampling_strategy=my_dict_2)
        rus_Group_3 = NearMiss(version=2, sampling_strategy=my_dict_3)
        rus_Group_4 = NearMiss(version=2, sampling_strategy=my_dict_4)
        rus_Group_5 = NearMiss(version=2, sampling_strategy=my_dict_5)
        
        X_resampled_Group_1, y_resampled_Group_1 = rus_Group_1.fit_resample(X_Group_1, y_Group_1)
        X_resampled_Group_2, y_resampled_Group_2 = rus_Group_2.fit_resample(X_Group_2, y_Group_2)
        X_resampled_Group_3, y_resampled_Group_3 = rus_Group_3.fit_resample(X_Group_3, y_Group_3)
        X_resampled_Group_4, y_resampled_Group_4 = rus_Group_4.fit_resample(X_Group_4, y_Group_4)
        X_resampled_Group_5, y_resampled_Group_5 = rus_Group_5.fit_resample(X_Group_5, y_Group_5)
        
        #change back the labels of Group_5
        y_resampled_Group_5 = y_resampled_Group_5.replace(favorable_label, unfavorable_label)
        
        #concat downsampled groups
        X_concat = pd.concat([X_resampled_Group_1, X_resampled_Group_2, X_resampled_Group_3, X_resampled_Group_4, X_resampled_Group_5])
        y_concat = pd.concat([y_resampled_Group_1, y_resampled_Group_2, y_resampled_Group_3, y_resampled_Group_4, y_resampled_Group_5])
        X_concat[label_names[0]] = y_concat
        
        X_total = X_concat.drop_duplicates().copy()
        
        X_resampled = X_total.drop(columns=[label_names[0]])
        y_resampled = X_total[label_names[0]]

        #convert downsampled data to aif360 dataset
        dataset_after = X_resampled
        dataset_after[label_names[0]] = y_resampled

        aif360_after = aif360.datasets.BinaryLabelDataset(
            df=dataset_after,
            label_names=label_names,
            favorable_label=favorable_label,
            unfavorable_label=unfavorable_label,
            protected_attribute_names=protected_attribute_names,
            unprivileged_protected_attributes=unprivileged_protected_attributes,
            privileged_protected_attributes=privileged_protected_attributes)

        return aif360_after
