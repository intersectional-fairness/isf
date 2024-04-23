# How to run unittest

## 1. Download Test Datasets
    Download the COMPAS Dataset by referring to the README for the AIF 360 toolkit.

    Example commands.
    ```
    !curl -O https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
    ```

## 2. Store in AIF 360 install location
    Store the Dataset in the location where the AIF 360 toolkit is installed.

    Example commands.
    ```
    !mv compas-scores-two-years.csv $(python -c 'import aif360; print(aif360.__path__[0])')/data/raw/compas/compas-scores-two-years.csv
    ```

## 3. Run
    Run unittest.

    Example commands.
    ```
    cd <isf root dir>
    python3 -m tests.unit_test
    ```
