import numpy as np
from rgx import *

csv = np.genfromtxt('dataset/spiral.csv', delimiter=',')
data = csv[:,0:2]
target = csv[:,3:6].astype(int)
layers = [90, 80]

w, b = train_nn(data, target, epoch=10000, maxmse=0.09, learning_r=0.028, reg_lambda=0.01, h_layers=layers, desclearning=True)
#np.save('_weight', w)
#np.save('_bias', b)

w = np.load('_weight.npy')
b = np.load('_bias.npy')
predict = test_nn(data, target, w, b, h_layers=layers)
rpredict = (np.argmax(predict, axis=1) + 1)
print('accuracy:\t', performance_metric(csv[:,2],rpredict,select="accuracy"))
print('f1micro:\t', performance_metric(csv[:,2],rpredict,select="f1micro"))
print('f1macro:\t', performance_metric(csv[:,2],rpredict,select="f1macro"))

p = visualize_bound(data, rpredict.astype(int), w, b, h_layers=layers)
p.show()