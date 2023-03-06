import random

def rand(lower, upper):
    
    return (upper - lower) * random.random() + lower

def create_matrix(rows, columns, fill = .0):
    
    matrix = []
    
    for i in range(rows):
        
        matrix.append([fill] * columns)
        
    return matrix

def test():
    
    random_number = rand(1, 2)
    
    print(random_number)
    
    matrix = create_matrix(10, 2, 1.5)
    
    print(matrix)
    
if __name__ == "__main__":
    
    print([.5] * 4)
    
    test()
    
    

