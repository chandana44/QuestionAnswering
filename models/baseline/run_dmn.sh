#!/bin/bash

# Concatenates all args to a single variable
pyargs="$*"

module load cuda/8.0

python -u babi-dmn.py $pyargs
