
import numpy as np

#Four-noded quadrilateral
weights = np.array([[1],[1],[1],[1]])
points = np.array([
    [-1/np.sqrt(3), -1/np.sqrt(3)],
    [1/np.sqrt(3), -1/np.sqrt(3)],
    [1/np.sqrt(3),  1/np.sqrt(3)],
    [-1/np.sqrt(3),  1/np.sqrt(3)]])      


#Nine-noded quadrilateral
                
weights = np.array([
    [25/81],
    [40/81],
    [25/81],
    [40/81],
    [64/81],
    [40/81],
    [25/81],
    [40/81],
    [25/81]]        
)
        
points = np.array([
[-np.sqrt(3/5), -np.sqrt(3/5)],
[0,     -np.sqrt(3/5)],
[np.sqrt(3/5), -np.sqrt(3/5)],
[-np.sqrt(3/5),     0],
[0, 0],
[np.sqrt(3/5),     0],
[-np.sqrt(3/5), np.sqrt(3/5)],
[0,       np.sqrt(3/5)],
[np.sqrt(3/5),  np.sqrt(3/5)]            
])