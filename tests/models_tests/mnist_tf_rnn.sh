#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/tf_main.py

### validate and timing
python ${mainpy} --model tf_rnn --learning-rate 0.01 --validate --timing
