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

images_path=$1
output_path=$2
python3 "catr/predict.py" "--input_path" $images_path "--output_path" $output_path "--v" "v2"