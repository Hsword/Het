#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/tf_main.py

### validate and timing
python ${mainpy} --model tf_logreg --validate --timing
