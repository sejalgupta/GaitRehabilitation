import numpy as np
import pandas as pd
input_data_sl = []
data = pd.read_excel('../data/WBDSinfo.xlsx')
filenames = data['FileName']
leglength = data['LegLength']
mass = data['Mass']
speed = data['GaitSpeed(m/s)']
age = data['Age']
i = 1
j = 1
for j in range(1,43):
   for i in range(1,9):
       file_to_check = '../data/WBDS' + '%02d'%j + 'walkT0' + str(i) + 'mkr.txt'
       file_to_check_1 = '../data/WBDS' + '%02d'%j + 'walkT0' + str(i) + 'mkr.tx'
       for index, file_ in enumerate(filenames):
           if file_ == file_to_check or file_ == file_to_check_1:
               input_data_sl.append([age[index], leglength[index], mass[index], speed[index]])
input_data_sl = np.array(input_data_sl)
np.save("../InputDataSL.npy", input_data_sl)
