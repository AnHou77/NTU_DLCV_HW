#!/bin/bash

# get model
wget -O 'models/dann_target_svhn_original.pth' 'https://www.dropbox.com/s/b0kri238b15hes0/dann_target_svhn_improved.pth?dl=1'

if [ "$SHELL" = "/bin/bash" ]; then
echo "--------------------------------"
echo "| your login shell is the bash |"
echo "--------------------------------"
else
echo "--------------------------------------------"
echo "| your login shell is not bash but $SHELL |"
echo "--------------------------------------------"
fi

target_data_path=$1
target_domain_name=$2
output_path=$3
python3 "inference_p3.py" $target_data_path $target_domain_name $output_path "improved"