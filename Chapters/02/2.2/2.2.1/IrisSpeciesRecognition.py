from sklearn.datasets import load_iris

import jax.numpy as jnp

def load():
    
    iris = load_iris()
    
    data = jnp.float32(iris.data)
    targets = jnp.float32(iris.target)
    target_names = iris.target_names
    descriptions = iris.DESCR
    feature_names = iris.feature_names
    
    #print("data[: 5] = {}".format(data[: 5]))
    print("targets[: 5] = {}".format(targets))
    #print("target_names[: 5] = {}".format(target_names))
    #print("descriptions[: 5] = {}".format(descriptions))
    #print("feature_names[: 5] = {}".format(feature_names))
    
def main():
    
    load()
    
if __name__ == "__main__":
    
    main()