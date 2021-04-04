#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/main.py

### validate and timing resnet18
python ${mainpy} --model resnet18 --learning-rate 0.1 --validate --timing

### validate and timing resnet34
# python ${mainpy} --model resnet34 --learning-rate 0.1 --validate --timing
