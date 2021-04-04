#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/main.py

### validate and timing vgg16
python ${mainpy} --model vgg16 --learning-rate 0.01 --validate --timing

### validate and timing vgg19
# python ${mainpy} --model vgg19 --learning-rate 0.01 --validate --timing

### run in cpu
# python ${mainpy} --model vgg16 --learning-rate 0.01 --gpu -1 --validate --timing
