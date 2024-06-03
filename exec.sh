#!/bin/bash

source /home/wl446/miniconda3/bin/activate camtune
cd ~/CamTune

# -----------------------------------------------------------------------------------------------
n=3
reboot=0
export MY_TEST_ENV="on"
python camtune/tune.py -e uniform -b sysbench_test_110 -m machine_b -n $n -r $reboot -s 11126
unset MY_TEST_ENV

conda deactivate



