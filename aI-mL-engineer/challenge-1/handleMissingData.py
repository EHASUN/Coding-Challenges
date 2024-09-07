# importing pandas as pd
import pandas as pd

# importing numpy as np
import numpy as np

# dictionary of lists
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, 45, 56, np.nan],
        'Third Score':[np.nan, 40, 80, 98]}

# creating a dataframe from list
df = pd.DataFrame(dict)

# using isnull() function  
df.isnull()

# using notnull() function 
df.notnull()

# filling missing value using fillna()  
df.fillna(0)

# filling a missing value with
# previous ones  
df.fillna(method ='pad')

# filling  null value using fillna() function  
df.fillna(method ='bfill')

# using dropna() function    
df.dropna(how = 'all')

# using dropna() function     
df.dropna(axis = 1)