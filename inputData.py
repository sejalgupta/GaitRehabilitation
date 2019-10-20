import numpy as np
import pandas as pd
trainingData = []
for j in range(1,43):
   for i in range(1,9):
       try:
           data = pd.read_csv('../data/WBDS' + '%02d'%j + 'walkT0' + str(i) + 'ang.txt', sep='\t')
           leglength = 0.89
           time = data['Time']
           hip_angle = data['RHipAngleZ']
           knee_angle = data['RKneeAngleZ']
           ankle_angle = data['RAnkleAngleZ']
           trainingData.append([hip_angle, knee_angle, ankle_angle])
       except:
           pass
trainingData = np.array(trainingData)
trainingData = np.transpose(trainingData, [0, 2, 1])
print(trainingData.shape)
np.save("../TrainingData.npy", trainingData)
