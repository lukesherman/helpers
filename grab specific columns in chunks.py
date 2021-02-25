
# Process a huge CSV

import pandas as pd
import numpy as np
from ast import literal_eval
import pickle


chunksize = 10 ** 2 ## This sets the number of rows to grab.
count = 1

for chunk in pd.read_csv("google_image_features_cnn.csv", #filename
                          chunksize=chunksize,low_memory=False):
    
    ####
    ### PROCESS HERE
    ## EXAMPLE BEGIN
    chunk["feature_list"] = chunk["features"].apply(literal_eval) # read column as literal

    chunk.pop("features") #drop features column
    
    data_final.append(chunk) # stack the tables, if the extracted data is still large this will cause memory issues.

    print("Iteration" , count , " complete.")
    count += 1
    ## EXAMPLE END
    if i == 1:
        names = chunk.columns

final_np_array = np.vstack(data_final) #this just stacks the dataframes into a array (basically a matrix)

final_df = pd.DataFrame(final_np_array) #This turns the matrix into a pandas df
final_df.columns = names