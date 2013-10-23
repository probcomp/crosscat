Crosscat
--------------

CrossCat is a domain-general, Bayesian method for analyzing high-dimensional data tables. CrossCat estimates the full joint distribution over the variables in the table from the data, via approximate inference in a hierarchical, nonparametric Bayesian model, and provides efficient samplers for every conditional distribution. CrossCat combines strengths of nonparametric mixture modeling and Bayesian network structure learning: it can model any joint distribution given enough data by positing latent variables, but also discovers independencies between the observable variables.

A range of exploratory analysis and predictive modeling tasks can be addressed via CrossCat, including detecting predictive relationships between variables, finding multiple overlapping clusterings, imputing missing values, and simultaneously selecting features and classifying rows. Research on CrossCat has shown that it is suitable for analysis of real-world tables of up to 10 million cells, including hospital cost and quality measures, voting records, handwritten digits, and state-level unemployment time series.

# Installation

### VM

We provide a [VirtualBox VM](https://docs.google.com/file/d/0B_CtKGJ4pH2TX2VaTXRkMWFOeGM/edit?usp=drive_web) ([VM_README](https://github.com/mit-probabilistic-computing-project/vm-install-crosscat/blob/master/VM_README.md)) for small scale testing of CrossCat.

**Note**: The VM is only meant to provide an out-of-the-box usable system setup.  Its resources are limited and large jobs will fail due to memory errors.  To run larger jobs, increase the VM resources or install directly to your system.

### Local (Ubuntu)
CrossCat can be installed locally on Ubuntu systems with

    git clone https://github.com/mit-probabilistic-computing-project/crosscat.git
    sudo bash crosscat/scripts/install_scripts/install.sh
    cd crosscat && PYTHONPATH=$PYTHONPATH:$(pwd) make cython

Don't forget to add crosscat to your python path.  For bash, this can be accomplished by substituting the correct value for <CROSSCAT_DIR> and running

    cat -- >> ~/.bashrc <<EOF
    export PYTHONPATH=\$PYTHONPATH:<CROSSCAT_DIR>
    EOF

# Documentation


[Python Client](https://docs.google.com/file/d/0B_CtKGJ4pH2TdmNRZkhmamg5aVU/edit?usp=drive_web)

[C++ backend](https://docs.google.com/file/d/0B_CtKGJ4pH2TeVo0Zk5IT3V6S0E/edit?usp=drive_web)

# Example

dha\_example.py ([github](https://github.com/mit-probabilistic-computing-project/crosscat/blob/master/examples/dha_example.py)) is a basic example of analysis using CrossCat.  For a first test, run the following from inside the top level crosscat dir

    python crosscat/examples/dha_example.py --num_chains 2 --num_transitions 2


**Note**: the default argument values take a considerable amount of time to run and are best suited to a cluster.

# License

[Apache License, Version 2.0](https://github.com/mit-probabilistic-computing-project/crosscat/blob/master/LICENSE)
