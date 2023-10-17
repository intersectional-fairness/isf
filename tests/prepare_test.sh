#!/bin/sh
# List directories that python3 refers to for searching packages
list_path() {
    python3 << EOF
import sys
for d in sys.path:
    print(d)
EOF
}

patch_dir=$(dirname $0)

# Download and store datasets necessary for testing
#  Parameter:
#    - $1: directory to store the datasets
prepare_test() {
    d="$1"
    if ! [ -f ${d}/aif360/data/raw/compas/compas-scores-two-years.csv ]; then
        curl -O https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
        mv compas-scores-two-years.csv ${d}/aif360/data/raw/compas/compas-scores-two-years.csv
    else
        echo "Necessary csv file already downloaded: $(ls -l ${d}/aif360/data/raw/compas/compas-scores-two-years.csv)"
        echo  ''
    fi
}

# Locate the directory where AIF360 is installed and store the datasets there
for d in $(list_path); do
    if [ -d "$d" ] && [ -d "${d}/aif360" ]; then
        prepare_test "$d"
    fi
done
