import numpy as np

"""

How to use asterisks in Python?

"""

def exemplify():

    a = [1, 2]

    print(*a)
    print("--------------------------------------")

    """
    
    1 2
    --------------------------------------

    """

    l = [2, 6]
    array = np.arange(*l)

    print(array)
    print("--------------------------------------")

    l = (2, 6)
    array = np.arange(*l)

    print(array)
    print("--------------------------------------")

    """
    
    [2 3 4 5]
    --------------------------------------
    [2 3 4 5]
    --------------------------------------
    
    """

    d = {"x": 1, "y": 2}

    # Read the key of dictionary
    print(*d)
    print("--------------------------------------")

    """
    
    x y
    --------------------------------------

    """

    # Cluster the addtional values and to form a list
    a, b, *l = [2, 3, 4, 5]

    print("a = {}, b = {}, l = {}".format(a, b, l))
    print("--------------------------------------")

    """
    
    a = 2, b = 3, l = [4, 5]
    --------------------------------------
    
    """

    # Gather additional parameters
    def print_function(x, *parameters):

        print("x = {}, *parameters = {}".format(x, parameters))
        print("--------------------------------------")

        """
        
        x = 1, *parameters = (2, 3)
        --------------------------------------
        
        """

    print_function(1, 2, 3)

    # Error
    # print_function(x = 1, 2, 3, 4)

    def print_function1(x, y):

        print(x)
        print(y)
        print("--------------------------------------")

        """
        
        1
        2
        --------------------------------------
        
        """

    d = {"x": 1, "y": 2}

    # ** will fecth the keys and its values from dictionary, and fill out the parameters for function
    print_function1(**d)

    # Mingle the * and **
    def print_function2(x, *parameter1, **parameter2):

        print(x)
        print(parameter1)
        print(parameter2)
        print("--------------------------------------")

        """
        
        1
        (2, 3, 4, 5)
        {'a': 6, 'b': 7}
        --------------------------------------
        
        """

    print_function2(1, 2, 3, 4, 5, a = 6, b = 7)

def start():

    exemplify()

def main():

    start()

if __name__ == "__main__":

    main()
