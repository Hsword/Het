#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/main.py

### use momentum optimizer
# python ${mainpy} --model mlp --validate --timing --opt momentum

### use nesterov momentum optimizer
# python ${mainpy} --model mlp --validate --timing --opt nesterov

### use adagrad optimizer
# python ${mainpy} --model mlp --validate --timing --opt adagrad

# ### use adam optimizer
python ${mainpy} --model mlp --validate --timing --opt adam --learning-rate 0.01
