#!/bin/bash

if [ "$SHELL" = "/bin/bash" ]; then
echo "--------------------------------"
echo "| your login shell is the bash |"
echo "--------------------------------"
else
echo "--------------------------------------------"
echo "| your login shell is not bash but $SHELL |"
echo "--------------------------------------------"
fi

test_data_dir=$1
output_csv=$2

python3 "inference_p1.py" $test_data_dir $output_csv