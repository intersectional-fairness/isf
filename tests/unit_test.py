# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 Fujitsu Limited
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

import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from logging import CRITICAL, getLogger
from os import environ
# Suppress warnings that tensorflow emits
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from aif360.datasets import CompasDataset

from isf.core.intersectional_fairness import IntersectionalFairness
from isf.utils.common import classify, output_subgroup_metrics, convert_labels, create_multi_group_label
from tests.stream import MuteStdout


MODEL_ANSWER_PATH = './tests/result/'


class TestStringMethods(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def _read_modelanswer(self, s_result_singleattr, s_result_combattr):
        # load of model answer
        ma_singleattr_bias = pd.read_csv(MODEL_ANSWER_PATH + s_result_singleattr, index_col=0)
        ma_combattr_bias = pd.read_csv(MODEL_ANSWER_PATH + s_result_combattr, index_col=0)
        return ma_singleattr_bias, ma_combattr_bias

    def _comp_dataframe(self, df1, df2):
        try:
            assert_frame_equal(df1, df2)
        except AssertionError:
            return False
        return True

    def _pickup_result(self, df_singleattr, df_combattr):
        # load of model answer
        result_singleattr_bias = df_singleattr[['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy']]
        result_combattr_bias = df_combattr[['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy']]
        return result_singleattr_bias, result_combattr_bias

    def setUp(self):
        getLogger().setLevel(CRITICAL)

        # load test dataset
        self.dataset = CompasDataset()
        convert_labels(self.dataset)
        self.ds_train, self.ds_test = self.dataset.split([0.7], shuffle=False, seed=1)

    def test01_AdversarialDebiasing(self):
        s_algorithm = 'AdversarialDebiasing'
        s_metrics = 'DemographicParity'

        # test
        with MuteStdout():
            ID = IntersectionalFairness(s_algorithm, s_metrics)
            ID.fit(self.ds_train)
            ds_predicted = ID.predict(self.ds_test)

        group_protected_attrs, label_unique_nums = create_multi_group_label(self.dataset)
        g_metrics, sg_metrics = output_subgroup_metrics(self.ds_test, ds_predicted, group_protected_attrs)

        # pickup
        result_singleattr_bias, result_combattr_bias = self._pickup_result(g_metrics, sg_metrics)

        # load of model answer
        ma_singleattr_bias, ma_combattr_bias = self._read_modelanswer("test01_result_singleattr.csv",
                                                                      "test01_result_combattr.csv")

        assert_frame_equal(result_singleattr_bias, ma_singleattr_bias, atol=0.2)
        assert_frame_equal(result_combattr_bias, ma_combattr_bias, atol=0.2)

    def test02_EqualizedOdds(self):
        s_algorithm = 'EqualizedOddsPostProcessing'
        s_metrics = 'EqualizedOdds'

        # test
        ds_train_classified, threshold, _ = classify(self.ds_train, self.ds_train)
        ds_test_classified, _, _ = classify(self.ds_train, self.ds_test, threshold=threshold)

        ID = IntersectionalFairness(s_algorithm, s_metrics)
        ID.fit(self.ds_train, dataset_predicted=ds_train_classified, options={'threshold': threshold})
        ds_predicted = ID.predict(ds_test_classified)

        group_protected_attrs, label_unique_nums = create_multi_group_label(self.dataset)
        g_metrics, sg_metrics = output_subgroup_metrics(self.ds_test, ds_predicted, group_protected_attrs)

        # pickup
        result_singleattr_bias, result_combattr_bias = self._pickup_result(g_metrics, sg_metrics)

        # load of model answer
        ma_singleattr_bias, ma_combattr_bias = self._read_modelanswer("test02_result_singleattr.csv",
                                                                      "test02_result_combattr.csv")

        # assert
        assert_frame_equal(result_singleattr_bias, ma_singleattr_bias)
        assert_frame_equal(result_combattr_bias,   ma_combattr_bias)

    def test03_Massaging(self):
        s_algorithm = 'Massaging'
        s_metrics = 'DemographicParity'

        ID = IntersectionalFairness(s_algorithm, s_metrics)
        ID.fit(self.ds_train)
        ds_predicted = ID.transform(self.ds_train)

        group_protected_attrs, label_unique_nums = create_multi_group_label(self.dataset)
        g_metrics, sg_metrics = output_subgroup_metrics(self.ds_train, ds_predicted, group_protected_attrs)

        # pickup
        result_singleattr_bias, result_combattr_bias = self._pickup_result(g_metrics, sg_metrics)

        # load of model answer
        ma_singleattr_bias, ma_combattr_bias = self._read_modelanswer("test03_result_singleattr.csv",
                                                                      "test03_result_combattr.csv")

        # assert
        assert_frame_equal(result_singleattr_bias, ma_singleattr_bias)
        assert_frame_equal(result_combattr_bias, ma_combattr_bias)

    def test04_RejectOptionClassification(self):
        s_algorithm = 'RejectOptionClassification'
        s_metrics = 'F1Parity'

        ds_train_classified, threshold, _ = classify(self.ds_train, self.ds_train)
        ds_test_classified, _, _ = classify(self.ds_train, self.ds_test, threshold=threshold)

        group_protected_attrs, label_unique_nums = create_multi_group_label(self.dataset)

        ID = IntersectionalFairness(s_algorithm, s_metrics,
                                    accuracy_metric='F1', options={'accuracy_metric_name': 'F1', 'metric_ub': 0.2, 'metric_lb': -0.2})
        ID.fit(self.ds_train, dataset_predicted=ds_train_classified)
        ds_predicted = ID.predict(ds_test_classified)

        g_metrics, sg_metrics = output_subgroup_metrics(self.ds_test, ds_predicted, group_protected_attrs)

        # pickup
        result_singleattr_bias, result_combattr_bias = self._pickup_result(g_metrics, sg_metrics)

        # load of model answer
        ma_singleattr_bias, ma_combattr_bias = self._read_modelanswer("test04_result_singleattr.csv",
                                                                      "test04_result_combattr.csv")

        # assert
        assert_frame_equal(result_singleattr_bias, ma_singleattr_bias)
        assert_frame_equal(result_combattr_bias, ma_combattr_bias)

    def test05_Massaging_AA(self):
        s_algorithm = 'Massaging'
        s_metrics = 'DemographicParity'

        debiasing_conditions = [{'target_attrs': {'sex': 1.0, 'race': 0.0}, 'uld_a': 0.8, 'uld_b': 1.2, 'probability': 1.0}]

        ID = IntersectionalFairness(s_algorithm, s_metrics,
                                    debiasing_conditions=debiasing_conditions, instruct_debiasing=True)
        ID.fit(self.ds_train)
        ds_predicted = ID.transform(self.ds_train)

        group_protected_attrs, _ = create_multi_group_label(self.dataset)
        g_metrics, sg_metrics = output_subgroup_metrics(self.ds_train, ds_predicted, group_protected_attrs)

        # pickup
        result_singleattr_bias, result_combattr_bias = self._pickup_result(g_metrics, sg_metrics)

        # load of model answer
        ma_singleattr_bias, ma_combattr_bias = self._read_modelanswer("test05_result_singleattr.csv",
                                                                      "test05_result_combattr.csv")

        # assert
        assert_frame_equal(result_singleattr_bias, ma_singleattr_bias)
        assert_frame_equal(result_combattr_bias, ma_combattr_bias)


if __name__ == "__main__":
    unittest.main()
