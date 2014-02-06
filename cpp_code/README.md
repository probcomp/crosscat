#CrossCat C++ Engine v2 (Proposal)

2/4/2014

## Purpose of revision
The main goals are encapsulation, readability and, as a result, extensibility. Adding functionality should require editing in as few places as possible. This reduces the difficulty of editing the code and writing tests. The second goal is speed. Searching and string-matching will be eliminated. Additionally, the engine revision is an opportunity to write comprehensive unit tests and to implement concurrency within the C++ engine (using [OpenMP](http://en.wikipedia.org/wiki/OpenMP)).

## Summary of proposed revisions
- Take advantage of C++11
    - `<random>` removes the need for Boost
- Exception handling with compiler flags (cython stops `assert` from working)
- Unit tests (and hooks) for all C++ functionality
- Move some sample and probability functionality from python to C++ (see [State](#state))
- Support for non-collapsed samplers. Allows support for any kind of data type you can think of.
    - **Note:** Uncollapsed samplers are currently implemented in BaxCat 
- The `Cluster` class will be removed; its functionality will be shared by `View` and a new class, `Feature`.
- Component model math will be moved from `Numerics` to private static functions in each component model
- Discontinue use of `std::map` within the inference engine. Hypers will be indexed with enums (see [ComponentModel](#componentmodel)). 
    - Discontinue use of strings as data in MultinomialComponentModel (replaced by ints). Python will be responsible for conversion of strings to ints.
    - Allows for deterministic reproduction of runs (assuming same seed and same number of processors)
- Parallelize operations
    - Discontinue use of iterators in these situations

# Structure

## Proposed class structure

State->View->Feature->ComponentModel

### State
The python engine will interact only with the state object. This removes the overhead associated with constructing component models from the metadata. Both `simple_predictive_probability` and `simple_predictive_sample` will be moved to `state`.

```python
state = State.p_State(M_c, T, X_L=X_L, X_D=X_D)
queries = [(0,1)]
constraints = [(2,1,3.2)]

# one sample from each query
samples = state.predictive_sample(queries)

# 100 samples from each query
samples = state.predictive_sample(queries, n=100)

# sample with contraints
samples = state.predictive_sample(queries, constraints)

# predictive probability
queries = [(0, 1, 2.1)]
logp = state.predictive_probability(queries)
```

`state` will have a `view` object for each view and a `feature` object for each column.

### View
Represents a crosscat view. Views contain vectors of `feature` objects. The view object will represent the concept of the cluster, or category, implicitly. The `cluster` object will no longer exist. Its functionality will be shared by the `feature` and the  `view`. The reasoning is that the bulk of the work the view does is the row transition, and the bulk of the work done by the row transition operation is calculating the probabilities of different rows of data under different clusters (component models). The row probability can be calculated by summing the log probabilities of the data in row r under the component models of cluster k, so the cluster does not need to be represented in the code and `cluster_lookup` is no longer needed. For example:

```cpp
void Feature::row_logp(unsigned int row, unsigned int cluster){
// Get the log likeihood of row under cluster 
    double lp = 0;
    for (int f = 0; f < num_features; f++){
        lp += features[f].data_logp(row, cluster);
    }
    return lp
}
```

### Feature
The purpose of feature is to represent a column in the table as a clustered data set. The feature will store the data, a vector of the hyperparameters,  and a vector of component model for each cluster. There will be a standard `feature` class for collapsed samplers and a subclass for uncollapsed samplers that also holds and updates a vector of component parameters used for proposing singleton clusters. 

### ComponentModel
All of a component model's functionality (minus utilities) will be moved to its object. Component models will share information about hyperparametrs and component parameters with their `feature` through vectors, rather than maps. Each component model must then have a specific enum to access the parameters in the vector in a readable way. For example

```cpp
void ContinuousComponentModel::update_hyperparams( const std::vector<double> &hypers){
    s = hypers[HYPER_S];
    m = hypers[HYPER_M];
    r = hypers[HYPER_R];
    nu = hypers[HYPER_NU];
}
```

The base `ComponentModel` will look something like this:

```cpp
class ComponentModel {
    
public:
    
    virtual void insert_element(double x);
    virtual void remove_element(double x);

    
    virtual double model_logp();
    // calulates the marginal likelihood (collapsed) or likelihood (uncollapsed)
    
    virtual double data_logp(double x);
    // calculates the probability of x in the given component model
    
    virtual double singleton_logp(double x);
    // calculates the prdictive porbability of x in a component by itself (collapsed)
    
    virtual double singleton_logp(double x, std::vector<double> &params);
    // calculates the likelihood of x in a component by itself. Draws coponent parameters for the
    // feature object to use if the singleton is accepted (uncollapsed)
    
    virtual void set_hypers(const std::vector<double> &hypers);
    // sets the hyperparameters of the component model
    
    virtual void set_params(const std::vector<double> &params);
    // sets the hyperparameters of the component model (uncollapsed)
    
    virtual std::vector<std::vector<double>> init_hyper_grids(const std::vector<double> &X, 
                                                                unsigned int n_grid);
    // returns a vector of vectors (grids) given the data in the feature and the
    // number of grid
    // points.
    
    virtual std::vector<double> update_hypers(const std::vector<ComponentModel> &cats,
                                              const std::vector<std::vector<double>> &hypers_grids);
    // Does the actural hyper transition. Calculates the conditionals and draws new hyperparameters
    // which are returned in a vector.
    
    // To send the suffstats and hypers to the python engine
    virtual std::map<std::string, double> get_suffstats_map();
    virtual std::map<std::string, double> get_hypers_map();
    
protected:
    
    unsigned int count; // number of data points assigned to this component model
    
};
```
Because the new engine will implement uncollapsed samplers, the names of the methods `predictive_logp` and `marginal_logp` are no longer accurate, so the more vague names `data_logp` and `model_logp` have been proposed.
The base class methods are all virtuals and will have no implementations. For increased encapsulation, component models will implement the bulk of their functionality in private static methods and look something like this:

```cpp

class NormalComponentModel : public ComponentModel {
    
    
public:
    // constructor with default arguments
    NormalComponentModel(unsigned int _N=0, double _sum_x=0, double _sum_x_sq=0,
                         double _m=0, double _r=1, double _s=1, double _nu=1);
    
    virtual void insert_element(double x) override;
    virtual void remove_element(double x) override;
    
    virtual double model_logp() override;
    virtual double data_logp(double x) override;
    virtual double singleton_logp(double x) override;
    
    virtual void set_hypers(const std::vector<double> &hypers) override;
    
    virtual std::vector<std::vector<double>> init_hyper_grids(const std::vector<double> &X,
                                                              unsigned int n_grid) override;
    
    virtual std::vector<double> update_hypers(const std::vector<ComponentModel> &cats,
                      const std::vector<std::vector<double>> &hypers_grids) override;
    
    virtual std::map<std::string, double> get_suffstats_map() override;
    virtual std::map<std::string, double> get_hypers_map() override;
    
private:
    // indexing enum
    enum hyper_idx {HYPER_M=0, HYPER_R=1, HYPER_S=2, HYPER_NU=3};
    
    // sufficient statistics
    // ---------------------
    double sum_x;
    double sum_x_sq;
    
    // hyper parameters
    // ----------------
    double m;
    double r;
    double s;
    double nu;
    
    // static helper functions
    // -----------------------
    static double calc_log_Z(double r, double s, double nu);
    // calculates the normalizing constant
    
    static void posterior_update_parameters(unsigned int N, double sum_x, double sum_x_sq,
                                     double &m, double &r, double &s, double &nu);
    // updates the hypers for the posterior, e.g. nu -> nu'
    
    static double calc_marginal_logp(unsigned int N, double sum_x, double sum_x_sq,
                              double m, double r, double s, double nu);
    // calculate marginal probability
    
    static double calc_predictive_logp(double x, unsigned int N, double sum_x, double sum_x_sq,
                              double m, double r, double s, double nu);
    // calculate predictive probability
    
    
    // calculate hyper conditionals
    // ----------------------------
    static std::vector<double> calc_hyper_m_conditional(const std::vector<ComponentModel> &cats,
                                                 const std::vector<double> &m_grid, unsigned int N,
                                                 double sum_x, double sum_x_sq,
                                                 double r, double s, double nu);
    static std::vector<double> calc_hyper_r_conditional(const std::vector<ComponentModel> &cats,
                                                 const std::vector<double> &r_grid, unsigned int N,
                                                 double sum_x, double sum_x_sq,
                                                 double m, double s, double nu);
    static std::vector<double> calc_hyper_s_conditional(const std::vector<ComponentModel> &cats,
                                                 const std::vector<double> &s_grid, unsigned int N,
                                                 double sum_x, double sum_x_sq,
                                                 double m, double r, double nu);
    static std::vector<double> calc_hyper_nu_conditional(const std::vector<ComponentModel> &cats,
                                                  const std::vector<double> &nu_grid, unsigned int N,
                                                  double sum_x, double sum_x_sq,
                                                  double m, double r, double s);
};
```

**Note:** `override` is [a feature of C++11](http://en.cppreference.com/w/cpp/language/override).

## Other files

### Numerics
`numerics` will no longer contain methods for each component class (these methods will be moved to their respective classes) but will be used for general, widely-used methods such as drawing an index from a vector of probabilities, or calculating the log CRP.

### Utils
`utils` will not change significantly, though it might be good to place it inside a namespace. May contain utility classes such as factories.

# Development 
There are two ways the development can proceed:
 1. Fresh start. Write everything from scratch.
    - Pros:
        - Faster--No writing temporary code to fit round pegs in square holes
    - Cons:
        - If the new engine produces different results than the old engine, it may be more difficult to pinpoint the cause.
 2. Incremental refactoring. Implement the redesign as a seriese of changes to the existing code.
    - Pros:
        - Leaves the code in a working state (reduces probability of failure)
    - Cons:
        - Slower requires thought on how to move toward an end state without breaking functionality

If development proceeds as incremental refactoring, the development will proceed in the following steps:

 1. Move numerics functionality to component models
    - Remove maps from multinomial
 2. Replace `Cluster` with `Feature`
    - Will require refactoring of `View`
 3. Add Non-collapsed `Feature` (`Feature_nc`)
 4. Move sample and probability functionality to `State`
 5. Add concurrency

 At each step, headers and full documentation will be provided for group review before proceeding with implementations.

