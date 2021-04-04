#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/main.py

### validate and timing
python ${mainpy} --model mlp --validate --timing

### run in cpu
# python ${mainpy} --model mlp --gpu -1 --validate --timing

### run with stream executor
# python ${mainpy} --model mlp --validate --timing --streams 1

### run with multi-stream executor
# python ${mainpy} --model mlp --validate --timing --streams 2
