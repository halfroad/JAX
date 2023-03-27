import jax

def indicator_function(i) -> bool:

    if i > 5:
        return 1
    else:
        return 0

if __name__ == "__main__":
    
    i = 10
    
    indicator = indicator_function(i)
    
    print("indicator = ", indicator)
             
             
