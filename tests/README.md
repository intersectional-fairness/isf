# How to run unittest

## 1. Download Test Datasets and store them in the AIF 360 installed location
    Download the COMPAS Dataset by referring to the README for the AIF 360 toolkit
    and store the Dataset in the location where the AIF 360 toolkit is installed.

    Example commands.
    ```
    cd <isf root dir>
    !./tests/prepare_test.sh
    ```

## 2. Run
    Run unittest.

    Example commands.
    ```
    cd <isf root dir>
    python3 -m tests.unit_test
    ```
