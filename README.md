Crosscat
--------------

CrossCat is a fully Bayesian nonparametric method for analyzing heterogenous, high dimensional data. It yields an estimate of the underlying full joint distribution, including the dependencies between variables, along with efficient samplers for every conditional distribution. The method is domain general, unsupervised and has no free parameters that require tuning. Thus, CrossCat provides a generic method for analyzing these data, where the output can be queried to answer an extremely wide array questions about the rows and attributes. Unlike standard statistical approaches, CrossCat does not make restrictive assumptions---such as linearity---and therefore provides reliable inferences without specific training in statistics.

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
