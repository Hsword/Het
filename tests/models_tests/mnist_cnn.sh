#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/main.py

### validate and timing
python ${mainpy} --model cnn_3_layers --validate --timing
