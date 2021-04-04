#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
mainpy=${workdir}/tf_main.py

### use momentum optimizer
# python ${mainpy} --model tf_mlp --validate --timing --opt momentum

### use nesterov momentum optimizer
# python ${mainpy} --model tf_mlp --validate --timing --opt nesterov

### use adagrad optimizer
# python ${mainpy} --model tf_mlp --validate --timing --opt adagrad

# ### use adam optimizer
python ${mainpy} --model tf_mlp --validate --timing --opt adam --learning-rate 0.01
