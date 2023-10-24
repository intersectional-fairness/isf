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

# Apply pathces necessary for ISF to codes in AIF360
#  Parameter
#    - $1: directory of installed AIF360 in which codes are to be modified.
apply() {
    d="$1"
    if ! [ -f $d/aif360/datasets/structured_dataset.py.orig ]; then
        patch -b $d/aif360/datasets/structured_dataset.py ${patch_dir}/structured_dataset.patch
    else
        echo "Patch already applied: $(ls -l $d/aif360/datasets/structured_dataset.py*)"
        echo  ''
    fi
    if ! [ -f $d/aif360/algorithms/postprocessing/reject_option_classification.py.orig ]; then
        patch -b $d/aif360/algorithms/postprocessing/reject_option_classification.py ${patch_dir}/reject_option_classification.patch
    else
        echo "Patch already applied: $(ls -l $d/aif360/algorithms/postprocessing/reject_option_classification.py*)"
        echo  ''
    fi
}

# Locate the directory where AIF360 is installed and apply the pathces to codes there
for d in $(list_path); do
    if [ -d "$d" ] && [ -d "${d}/aif360" ]; then
        apply "$d"
    fi
done
