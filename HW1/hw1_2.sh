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
output_dir=$2

python3 "inference_p2.py" $test_data_dir $output_dir