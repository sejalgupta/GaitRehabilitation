from SupervisedLearning import *
import math
NNmodel = NeuralNetwork()
test_data = [
       [50, 0.80, 65, 1.0],
       [50, 0.80, 65, 2.0]
   ]
l1 = 17
l2 = 15
l3 = 5
predicting = NNmodel.predict(test_data)
decoding = np.reshape(predicting, [-1, 101, 3])
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(2, 3, 1)
ax1 = fig.add_subplot(2, 3, 2)
ax2 = fig.add_subplot(2, 3, 3)
ax3 = fig.add_subplot(2, 3, 4)
ax4 = fig.add_subplot(2, 3, 5)
ax5 = fig.add_subplot(2, 3, 6)  
for i in range(predicting.shape[0]):
   hip_angle = decoding[i,:,0]
   ax.set_title("HIP ANGLE")
   ax.plot(hip_angle, label='%d'%i)
   knee_angle = decoding[i,:,1]
   ax1.set_title("KNEE ANGLE")
   ax1.plot(knee_angle, label='%d'%i)
   np.save("../KneeAngle.npy", knee_angle)
   knee = np.load("../KneeAngle.npy")
   np.savetxt("kneeAngle.csv", knee, delimiter=",")
   ankle_angle = decoding[i,:,2]
   ax2.set_title("ANKLE ANGLE")
   ax2.plot(ankle_angle, label='%d'%i)
   x = []
   y = []
   a = []
   b = []
   c = []
   d = []
   for i in range(0,99):
       alpha = 270 + hip_angle[i]
       beta = alpha - knee_angle[i]
       gamma = beta + 90 + ankle_angle[i]
       x.append(l1*math.cos(math.radians(alpha)))
       y.append(l1*math.sin(math.radians(alpha)))
       a.append(x[i]+(l2 * math.cos(math.radians(beta))))
       b.append(y[i]+(l2 * math.sin(math.radians(beta))))
       c.append(a[i]+(l3* math.cos(math.radians(gamma))))
       d.append(b[i] +(l3* math.sin(math.radians(gamma))))
   ax3.set_title("HIP POSITION")
   ax3.plot(x, y)
   ax4.set_title("KNEE POSITION")
   ax4.plot(a, b)
   ax5.set_title("ANKLE POSITION")
   ax5.plot(c, d,)
plt.show()
