#! /bin/bash

# along with the !# add the following line(s) to .git/hooks/pre-commit
python -m unittest discover unit_tests '*.py'