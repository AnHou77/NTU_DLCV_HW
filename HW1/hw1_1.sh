#!/bin/bash

wget -O './resnet152_0.8740.pth' 'https://www.dropbox.com/s/y5fd5nf1u1wta7l/resnet152_0.8740.pth?dl=1'

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