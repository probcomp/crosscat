import random

import crosscat.tests.component_model_extensions.ContinuousComponentModel as ccmext


def test_constructor(component_model_class):
    N = 10
    m = component_model_class.from_parameters(N)
    
def test_draw_component_model_params(component_model_class):
    component_model = component_model_class.from_parameters(N)
    draw = component_model_class.sample_parameters_given_hyper()
    
    assert type(draw) is dict
    
    model_parameter_bounds = component_model_class.get_model_parameter_bounds()
    
    for key, value in draw.itervalues():
        try:
            bounds = model_parameter_bounds[key]
        except IndexError:
            raise IndexError("%s is not a valid parameter.")
            
        if value <= bounds[0] or value >= bounds[1]:
            raise ValueError("%s should be in %s" % (key, str(bounds)))
            
def test_draw_component_model_hyperparameters(component_model_class):
    
    component_model = component_model_class.from_parameters(N)
    
    random.seed(0)
    X = [random.normalvariate(data_mean, data_std) for i in range(N)]
    
    draw = component_model_class.draw_hyperparameters(X)
    
    hyperparameter_bounds = component_model_class.get_hyperparameter_bounds()
    
    for key, value in draw.itervalues():
        try:
            bounds = model_parameter_bounds[key]
        except IndexError:
            raise IndexError("%s is not a valid parameter.")
            
        if value <= bounds[0] or value >= bounds[1]:
            raise ValueError("%s should be in %s" % (key, str(bounds)))
            
