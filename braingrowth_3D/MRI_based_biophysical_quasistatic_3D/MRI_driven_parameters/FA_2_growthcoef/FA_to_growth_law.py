import math

def FA_to_tangential_growth_law_linearrelationship(linear_coef, normalized_fa_nodal_value):
    # linear
    alphaTAN_nodal_value = linear_coef * normalized_fa_nodal_value
    
    return alphaTAN_nodal_value


def FA_to_tangential_growth_law_exponential(linear_coef, normalized_fa_nodal_value):
    """
    Supposing:
    FA(tGW) = a.exp(-exp(-b(tGW-c))) (1) --> Gompertz to fit with dhcp MRI FA data
    alphaTAN = 0.2 * tGW (2) in our model 
    
    We obtain, incorporating (2) into (1): FA(tGW) = A exp(-(t - t0)/tau) + B
    
    Finally applyin ln, we get the below formula for alphaTAN = f(FA)
    """
    
    # alphaTAN = linear_coef * tGW
    # FA(tGW) = A exp(-(tGW - t0)/tau) + B 
    # => FA(tGW) = A exp(-(alphaTAN/linear_coef - t0)/tau) + B
    
    # ln(FA - b) = lnA - (alphaTAN/linear_coef - t0)/tau
    # => lnA - ln(FA - b) = (alphaTAN/linear_coef - t0)/tau
    # => alphaTAN = linear_coef * ( tau*(lnA - ln(FA - b)) + T0 ) 
    
    a = 2.
    b = 0.0
    T0 = 22
    tau = 1.
    alphaTAN_nodal_value = linear_coef * ( tau *(math.log(a) - math.log(normalized_fa_nodal_value - b)) + T0 ) 
    
    return alphaTAN_nodal_value


def FA_to_tangential_growth_law_gompertz(linear_coef, normalized_fa_nodal_value):
    """
    Supposing:
    FA(tGW) = a.exp(-exp(-b(tGW-c))) (1) --> Gompertz to fit with dhcp MRI FA data
    alphaTAN = 0.2 * tGW (2) in our model 
    
    We obtain, incorporating (2) into (1): FA(tGW) = a.exp(-exp(-b(alphaTAN/0.2-c)))
    
    Finally applyin ln, we get the below formula for alphaTAN = f(FA)
    """
    
    a = 2.
    b = 0.2
    c = 20. # Gompertz law parameters
    alphaTAN_nodal_value = linear_coef * ( (- math.log( math.log(a) - math.log(normalized_fa_nodal_value) ) ) / b + c )
    
    return alphaTAN_nodal_value