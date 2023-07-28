from typing import List, Optional, Iterable, TypeVar, Union
import pandas as pd
import unittest


T = TypeVar('T')


class TestCaseExtension:
    def __init__(self, testCase: unittest.TestCase):
        self.testCase = testCase

    def assertAlmostEqualIter(
            self, iter1: Iterable[T], iter2: Iterable[T], places: Optional[int] = 7, delta: Optional[Union[float, int]] = None, raw=False, n_check: int = 3):
        """Check each pair of values in `iter1` and `iter2` with `unittest.TestCase.assertAlmostEqual`"""
        prec = 7 if places is None else places
        more_check_required = n_check
        differed = []
        for idx, (v1, v2) in enumerate(zip(iter1, iter2)):
            try:
                self.testCase.assertAlmostEqual(v1, v2, places=places, delta=delta)
            except AssertionError:
                differed.append((idx, (round(v1, prec), round(v2, prec))))
                more_check_required -= 1
                if more_check_required <= 0:
                    break
        diff_count = len(differed)
        if diff_count == 0:
            return
        rest = '...' if diff_count > n_check else ''
        differed_str = f"{differed[:n_check]}{rest}"
        if raw:
            return differed_str
        NL = '\n    '
        raise AssertionError(
            f"The following items are different (position, values):{NL}{differed_str}")

    @staticmethod
    def _diff_str(idx_expr):
        return f"{idx_expr[0]}: {idx_expr[1]}"

    def assertAlmostEqualDF(
            self, df1: pd.DataFrame, df2: pd.DataFrame, columns: List[str],
            places: Optional[int] = 7, delta: Optional[Union[float, int]] = None, n_check: int = 3):
        """Check each pair of columns in `df1` and `df2` with `assertAlmostEqual`"""
        more_check_required = n_check
        diffs = []
        for name in columns:
            differed_str = self.assertAlmostEqualIter(
                iter1=df1[name], iter2=df2[name], places=places, delta=delta, raw=True)
            if not differed_str:
                continue
            diffs.append((name, differed_str))
            more_check_required -= 1
            if more_check_required <= 0:
                break
        diffs_count = len(diffs)
        if diffs_count:
            NL = '\n    '
            rest = f'{NL}...' if diffs_count > n_check else ''
            raise AssertionError(
                f"The following rows are different (position, rows):{NL}"
                f"{NL.join(map(self._diff_str, diffs[:n_check]))}"
                f"{rest}"
            )
