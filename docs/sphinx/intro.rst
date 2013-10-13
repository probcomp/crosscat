Intro
=====

What is CrossCat?
-----------------

CrossCat is a fully Bayesian nonparametric method for analyzing heterogenous, high dimensional data. It yields an estimate of the underlying full joint distribution, including the dependencies between variables, along with efficient samplers for every conditional distribution. The method is domain general, unsupervised and has no free parameters that require tuning. Thus, CrossCat provides a generic method for analyzing these data, where the output can be queried to answer an extremely wide array questions about the rows and attributes. Unlike standard statistical approaches, CrossCat does not make restrictive assumptions---such as linearity---and therefore provides reliable inferences without specific training in statistics.

Why CrossCat?
-------------

The ability to analyze large, heterogenous high-dimensional datasets is critical across fields. Typically these datasets take the form of tables with rows representing entities and columns representing entity attributes and can be as large as millions of rows and hundreds of columns. One such data source may be a medical records database, the entities are individual persons and their attributes are health/medical related information about them such as height, weight, bmi, common complaints, etc. One may be interested querying such data to get answers regarding which attributes are dependent, which rows are similar, likely values of unobserved cells or probable but not-previously-observed instances.  CrossCat provides all these functions and more.

