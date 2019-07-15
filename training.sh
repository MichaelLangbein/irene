#! /bin/bash
conda activate tensorflow
python training.py 2>&1 | tee trainingOutput.txt