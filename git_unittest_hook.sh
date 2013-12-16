#! /bin/bash

# along with the !# add the following line(s) to .git/hooks/pre-commit
python -m unittest discover crosscat/tests/unit_tests '*.py'