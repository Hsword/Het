#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/tf_main.py

### validate and timing
python ${mainpy} --model tf_mlp --validate --timing

### run in cpu
# python ${mainpy} --model tf_mlp --gpu -1 --validate --timing
