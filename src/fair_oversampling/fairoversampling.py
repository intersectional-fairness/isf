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

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:09:41 2022

@author: sonoda.ryosuke
"""

import numpy as np
from scipy import stats
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
import itertools


def preprocess(X, Y, A):
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int8)
    A = np.array(A, dtype=np.int8)
    ### create cluster by classes and attributes
    def make_cluster(X, Y, A):
        num_class = len(set(Y))
        num_group = len(set(A))
        cluster_Indices = [[] for _ in range(num_class)]
        cluster_sizes = [[] for _ in range(num_class)]
        for y in range(num_class):
            for a in range(num_group):
                index = np.where(np.logical_and(Y == y, A == a))[0]
                cluster_Indices[y].append(index)
                cluster_sizes[y].append(len(index))
        return cluster_Indices, cluster_sizes, num_class, num_group

    cluster_Indices, cluster_sizes, num_class, num_group = make_cluster(X, Y, A)
    M = np.max(cluster_sizes)
    D = X.shape[1]
    y_majority = np.where(M == cluster_sizes)[0][0]
    a_majority = np.where(M == cluster_sizes)[1][0]
    a_minority = np.where(np.min(cluster_sizes) == cluster_sizes)[1][0]
    y_minority = 1 if y_majority == 0 else 0

    return X, Y, A, cluster_Indices, num_class, num_group, M, D

def local_density(X, Y, num_class, num_group, cluster_Indices, k):
    ### the k nearest neighbors density ###
    nn_ = NearestNeighbors(n_neighbors=k + 1)
    nn_.fit(X)
    nn_dis, nn_num = nn_.kneighbors(X)
    nn_dis_, nn_num_ = nn_dis[:, 1:], nn_num[:, 1:]
    return nn_dis_, nn_num_
    
class FairOverSampler:
    """
    The FairOversampling technique is based on the following paper

    Author: Ruysuke Sonoda<br>
    Title: Fair oversampling technique using heterogeneous clusters<br>
    Affiliation: Fujitsu Ltd., 4-1-1 Kamikodanaka, Nakahara-ku, Kawasaki-shi, 211-8588, Kanagawa, Japan

    and can be downloaded from<br>
    https://www.sciencedirect.com/science/article/abs/pii/S0020025523006448<br>
    (see also https://arxiv.org/abs/2305.13875).

    Parameters
    ----------
    alpha: float
        Hyperparameter of the fair oversampling algorithm.  Oversampling is carried out by mixing in heterogeneous classes or heterogeneous groups, while the alpha controls the probabilitiy of mixing. A larger alpha results in more mixing between heterogeneous classes, while the reverse results in more mixing between heterogeneous groups. The range is unlimited, but an initial value of 1 is a good starting point.  Please, refer to the paper for more details.
    k: Integer
        number of nearest neighbours
    random_state : Integer
        random state to ensure reproducibility

    Returns
    ----------
    dataset:
        Oversampled dataset
    """

    def __init__(self, alpha:float, k=5, random_state=0):
        """
        it constructs a new FairDownSampler object
        """
        self.alpha = alpha
        #self.protected_idx = protected_idx
        self.k = k  # 10
        self.random_state = check_random_state(random_state)
        self.eps = 1e-10
    

    def fit_resample(self, X, Y, A):
        """
        It oversamples a given dataset

        Parameters
        ----------
        X : numpy array
            the features of the input dataset
        Y: numpy array
            the labels of the input dataset
        A: numpy array
            an array that represents the different intersectional groups
            
            
        Returns
        ----------
        dataset:
            Oversampled aif360 dataset

        """
        X, Y, A, cluster_Indices, num_class, num_group, M, D = preprocess(X, Y, A)
        self.num_group = num_group
        nn_dis_, nn_num_ = local_density(
            X, Y, num_class, num_group, cluster_Indices, self.k
        )
        ### generation synthetic samples ###
        new_features, new_classes, new_groups = (
            np.empty((0, D)),
            np.empty(0),
            np.empty(0),
        )
        
        for y in range(num_class):
            for a in range(num_group):
                to_samples = M - len(cluster_Indices[y][a])
                X_s = self.selective_smote(
                    X, Y, cluster_Indices, y, a, nn_num_, to_samples, nn_dis_
                )
                new_features = np.r_[new_features, X_s]
                new_classes = np.r_[new_classes, np.repeat(y, to_samples)]
                # new_groups = np.r_[new_groups, np.repeat(a, to_samples)]

        return (
            np.concatenate([X, new_features]),
            np.concatenate([Y, new_classes]),
            # np.concatenate([A, new_groups]),
        )
        
    def select_weight(self, cluster_Indices, y, _y, a):
        H_y = len(cluster_Indices[_y][a])
        H_a = 0
        for _a in range(self.num_group):
            if _a != a:
                H_a += len(cluster_Indices[y][_a])
                
        return (H_y / (H_y + H_a)) ** self.alpha

    def selective_smote(
        self, X, Y, cluster_Indices, y, a, nn_num_, to_samples, nn_dis_
    ):
        _y = 1 if y == 0 else 0
        s = self.select_weight(cluster_Indices, y, _y, a)

        to_samples_Y = int(to_samples * s)
        to_samples_A = to_samples - to_samples_Y

        Indices_y = self.fair_choice(cluster_Indices[y], a, to_samples_A, nn_num_, Y, y)

        X_n = self.neutr_smote(
            X,
            Y,
            y,
            cluster_Indices[y][a],
            # idx[:to_samples_A]
            Indices_y,
            nn_num_,
            to_samples_A,
            nn_dis_,
        )
        X_m = self.mixup_smote(
            X,
            Y,
            y,
            cluster_Indices[y][a],
            # idx[to_samples_A:],
            cluster_Indices[_y][a],
            nn_num_,
            to_samples_Y,
            nn_dis_,
        )

        return np.concatenate([X_n, X_m], axis=0)

    def mixup_smote(self, X, Y, y, Indices, Jndices, nn_num_, to_samples_Y, nn_dis_):
        ### i selection ###
        idx = self.random_state.choice(Indices, to_samples_Y, replace=True)
        ### j selection ###
        jdx = self.random_state.choice(Jndices, to_samples_Y, replace=True)
        ### interpolation weight ###
        d = (np.max(nn_dis_[idx], axis=1) / (np.linalg.norm(X[jdx] - X[idx], axis=1) + self.eps))[:, np.newaxis]
        density = np.array([np.sum(Y[nn_num_[i]] == y) for i in idx]) / self.k
        w = self.random_state.uniform(0, density)[:, np.newaxis]
        X_new = X[idx] + w * (X[jdx] - X[idx]) * d
        return X_new.astype(X.dtype)

    def neutr_smote(self, X, Y, y, Indices, Jndices, nn_num_, to_samples_A, nn_dis_):
        ### i selection ###
        idx = self.random_state.choice(Indices, to_samples_A, replace=True)
        ### j selection ###
        jdx = Jndices
        ### interpolation weight ###
        d = (np.max(nn_dis_[idx], axis=1) / (np.linalg.norm(X[jdx] - X[idx], axis=1) + self.eps))[:, np.newaxis]
        density = np.array([np.sum(Y[nn_num_[i]] == y) for i in idx]) / self.k
        w = self.random_state.uniform(0, density)[:, np.newaxis]
        X_new = X[idx] + w * (X[jdx] - X[idx]) * d
        ### fix the protected attributes ###
        return X_new.astype(X.dtype)
    
    def fair_choice(self, cluster_Indices_y, a, to_samples_A, nn_num_, Y, y):
        Indices_y = cluster_Indices_y.copy()
        Indices_y.pop(a)
        
        num_A = np.array([len(i) for i in Indices_y]) ** self.alpha
        weight = ((1 / num_A) / np.sum(1 / num_A) * to_samples_A).astype(int)

        fair_Indices_y = []
        for a,Indices in enumerate(Indices_y):
            n_same = np.array([np.sum(Y[nn_num_[i]] == y) for i in Indices])
            b = np.logical_and(n_same >= 0, n_same < self.k / 2)
            fair_Indices_y.append(np.random.choice(Indices, weight[a]+1, replace=True))
            
        L = list(itertools.chain.from_iterable(fair_Indices_y))       
        return np.random.choice(L, to_samples_A, replace=False)