import numpy as np
import pandas as pd

rand_generator =np.random.default_rng()

def generate_values(n_features, n_values):
    features=rand_generator.random((n_features, n_values))
    weights=rand_generator.random((1,n_values))[0]
    target =np.random.choice([0,1],n_features)
    data=pd.DataFrame(features,columns=["x0","x1","x2"])
    data["targets"]=target
    return data,weights


def get_weight_sum(features,weights,bias):
    return np.dot(features,weights)+bias

def sigmoid_function(w_sum):
    return 1/(1+np.exp(-w_sum))

def cross_entropy(target,prediction):
    return -(target * np.log10(prediction) + (1 - target) * np.log10(1 - prediction))


def gradient_descent(x, y, weights, bias, learnrate, pred):
    new_weights=[]
    bias+=learnrate*(y-pred)
    for element,weight in zip(x,weights):
        new_weight=weight+learnrate*(y-pred)*element
        new_weights.append(new_weight)
        print (element,new_weight)

    return new_weights,bias