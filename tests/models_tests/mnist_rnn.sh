#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/main.py

### validate and timing
python ${mainpy} --model rnn --learning-rate 0.01 --validate --timing
