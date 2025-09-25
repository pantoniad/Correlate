import pandas as pd
import numpy as np
import sys 
import os
#sys.path.append("Classes")
#sys.path.append(os.path.join(os.path.dirname(__file__), "Classes"))
from Classes.data_processor_class import data_process

diction = {
    "Column 1": {
        "Value 1": 1,
        "Value 2": 2,
        "Value 3": 3
    },
    "Column 2":{
        "Value 1": 4, 
        "Value 2": 5,
        "Value 3": 6
    },
    "Column 3":{
        "Value 1": 7, 
        "Value 2": 8,
        "Value 3": 9
    }
}

df = pd.DataFrame(
    data = diction,
    index = ["Value 1", "Value 2", "Value 3"]
)

# Test 1: Only df. Expecting the same dataframe as a return
df_final1 = data_process.df_former(df)
print(df_final1)

# Test 2: No rows, no parameter. Expecting only column 1
clmns = ["Column 1"]
df_final2 = data_process.df_former(df, clmns = clmns)
print(df_final2)

# Test 3: No rows, with parameter. Expecting only columns 1 and 2
clmns = ["Column 1", "Column 2"]
parameters = "3"
df_final3 = data_process.df_former(df, clmns = clmns, parameter = parameters)
print(df_final3)

# Test 4: no columns, no parameter. Expecting row 1 and 2 for all columns
rows = np.array([0, 2])
df_final4 = data_process.df_former(df, rows = rows)
print(df_final4)

# Test 5: no columns, with parameter. Expecting row 1 and column 3
rows = np.array([0, 2])
parameters = "3"
df_final5 = data_process.df_former(df, rows = rows, parameter = parameters)
print(df_final5)