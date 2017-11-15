Crosscat
========

.. image:: https://travis-ci.org/probcomp/crosscat.svg?branch=master
    :target: https://travis-ci.org/probcomp/crosscat

CrossCat is a domain-general, Bayesian method for analyzing high-dimensional data tables. CrossCat estimates the full joint distribution over the variables in the table from the data, via approximate inference in a hierarchical, nonparametric Bayesian model, and provides efficient samplers for every conditional distribution. CrossCat combines strengths of nonparametric mixture modeling and Bayesian network structure learning: it can model any joint distribution given enough data by positing latent variables, but also discovers independencies between the observable variables.

A range of exploratory analysis and predictive modeling tasks can be addressed via CrossCat, including detecting predictive relationships between variables, finding multiple overlapping clusterings, imputing missing values, and simultaneously selecting features and classifying rows. Research on CrossCat has shown that it is suitable for analysis of real-world tables of up to 10 million cells, including hospital cost and quality measures, voting records, handwritten digits, and state-level unemployment time series.

Installation
===============
Local (Ubuntu)
--------------

You can install CrossCat using pip (no need to clone from git)::

$ pip install crosscat

If you'd like to install from source, CrossCat can be successfully installed locally on bare Ubuntu server 14.04 systems with::

    $ sudo apt-get install build-essential cython python
    $ sudo apt-get install python-setuptools python-numpy
    $ git clone https://github.com/probcomp/crosscat.git

    $ cd crosscat
    $ python setup.py build
    $ python setup.py install  # or python setup.py develop

CrossCat can also be installed in a local Python virtual environment: ::

    $ cd crosscat
    $ virtualenv --system-site-packages /path/to/venv
    $ . /path/to/venv/bin/activate
    $ python setup.py build
    $ python setup.py install  # or python setup.py develop

A similar process has been found to work on OSX.

Tests
=====

To run the automatic tests:

    $ ./check.sh

Documentation
=============

**Note**: The VM is only meant to provide an out-of-the-box usable system setup.  Its resources are limited and large jobs will fail due to memory errors.  To run larger jobs, increase the VM resources or install directly to your system.


`Python Client`_

.. _`Python Client`: https://docs.google.com/file/d/0B_CtKGJ4pH2TdmNRZkhmamg5aVU/edit?usp=drive_web)

`C++ backend`_

.. _`C++ backend`: https://docs.google.com/file/d/0B_CtKGJ4pH2TeVo0Zk5IT3V6S0E/edit?usp=drive_web)

Example
========

dha\_example.py (github_) is a basic example of analysis using CrossCat.  For a first test, run the following from above the top level crosscat dir

.. _github: https://github.com/probcomp/crosscat/blob/master/examples/dha_example.py

::

    python crosscat/examples/dha_example.py crosscat/www/data/dha.csv --num_chains 2 --num_transitions 2


**Note**: the default argument values take a considerable amount of time to run and are best suited to a cluster.

License
=======

`Apache License, Version 2.0`_

.. _`Apache License, Version 2.0`: https://github.com/probcomp/crosscat/blob/master/LICENSE)
