import numpy as np
def calcul_fair(values):
    '''
    @brief:
        calculate fairness
    @params:
        values: a list of values
    '''
    values = np.array(values)
    n = len(values)
    if sum(values) != 0.:
        return pow(sum(values), 2)/(n*sum(pow(values, 2)))
    else:
        return 1.
    
#values = [0,5,100]
#values = [0,7,100]
values = [1, 3.3]
print(calcul_fair(values))