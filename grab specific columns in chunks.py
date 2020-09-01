
# Make a new csv
# For each line of the old CSV:  Take all the 'fields' from the old csv
# Write to new csv

import pandas as pd
import numpy as np


fields = np.array([1,2,5,6,7,51,54,57,58,104,136,149,215,274,281,294,295,298])-1 #set of columns that I want from the csv file (-1) is there because I originally tried to do this in R

data_final = []

chunksize = 10 ** 4 #this is just some file size, could be smaller or lower

count = 1
for chunk in pd.read_csv("file.txt", #filename
                         header=0, encoding="unicode_escape", chunksize=chunksize,low_memory=False):
    data = chunk.get_chunk(chunksize)[:, fields]
    data = chunk[chunk.columns[fields]]
    data_final.append(data) # stack the tables, if the extracted data is still large this will cause issues.
    #You could add lines here so that you ONLY grab data rows that match your filters. That would dramatically reduce the memory usage.

    print("Iteration" , count , " complete.")
    count += 1

final_np_array = np.vstack(data_final) #this just stacks the dataframes into a array (basically a matrix)

final_Pandas_DF = pd.DataFrame(final_np_array) #This turns the matrix into a pandas df

correct_column_names = chunk.columns[fields] #make sure we get the column names back
final_Pandas_DF.columns = [correct_column_names] #set the column names

#pickle.dump(final_Pandas_DF, open( "output_file.p", "wb" ) ) #This is to save the reduced data frame as a pickle (python thing) so you dont have to run this again
#final_Pandas_DF.to_csv('output_file.csv') #This is to save as a csv so you dont have to run this all again.

