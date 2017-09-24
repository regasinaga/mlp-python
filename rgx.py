import numpy as np
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
colors = np.array(['#ff3333', '#ff6633','#ff9933',
					'#ffcc33', '#ffff33','#ccff33',
					'#99ff33', '#33ff33', '#33ff99',
					'#33ffff', '#3399ff', '#3333ff',
					'#9933ff', '#ff33ff', '#ff3366'])
def recall(param):
	return param[3] / (param[3] + param[1] + 1)

def precision(param):
	return param[3] / (param[3] + param[0] + 1)

def confmatrix(actual, predicted):
	actual = actual.astype(int)
	predicted = predicted.astype(int)
	
	ts = len(np.unique(actual))
	matrix = np.zeros((ts,ts))
	rows = len(predicted)
	for r in range(0,rows):
		if(predicted[r]-1 < ts):
			matrix[actual[r]-1, predicted[r]-1] += 1

	return matrix

def visualize_bound(d, t, w, b,h_layers=[2], title="", preprocess=1):
	global colors

	# preparing data
	min_values = np.min(d[:,:], axis=0)
	max_values = np.max(d[:,:], axis=0)

	difference = 0.25
	attr1 = np.arange(min_values[0], max_values[0], difference)
	attr2 = np.arange(min_values[1], max_values[1], difference)
	mark = 'o'

	# iterate
	tn = len(np.unique(t))
	pd = np.zeros((len(attr1)*len(attr2),2))
	fake = np.zeros((len(attr1)*len(attr2),2))
	c = len(attr2)
	for y in range(0,len(attr2)):
		for x in range(0,len(attr1)):
			pd[x*c+y,0] = attr1[x]
			pd[x*c+y,1] = attr2[y]

	transfpd = pd * preprocess
	predict = test_nn(transfpd, fake, w, b, h_layers=h_layers)
	predict = np.argmax(predict, axis=1) + 1

	for tc in range(0, tn):

		index = np.where(predict==tc+1)
		mark = 'x' if tc % 2 == 0 else 'o'
		plot.scatter(pd[index,0], pd[index,1],
							marker=mark, color=colors[tc], label='kelas '+str(tc+1))
	plot.title(title)
	return plot

def performance_metric(t, p, select='accuracy'):
	if select=='accuracy':
		rows = len(t)
		correct = np.sum(p - t == 0)
		return correct / rows
	elif select=='f1micro':
		matrix = confmatrix(t,p)
		target_n = len(np.unique(t))
		param = np.zeros((target_n,4))

		precisions = np.zeros(target_n)
		recalls = np.zeros(target_n)
		until = target_n - 1
		for tc in range(0, target_n):
			after = np.min((tc+1,target_n - 1))
			# calculate true positive t
			param[tc,3] = matrix[tc,tc]
			# calculate true negative t
			param[tc,2] = np.sum(matrix[0:tc,0:tc]) + np.sum(matrix[0:tc,after:until]) + np.sum(matrix[after:until,0:tc]) + np.sum(matrix[after:until,after:until])
			# calculate false positive t
			param[tc,1] = np.sum(matrix[0:tc,tc]) + np.sum(matrix[after:until,tc])
			# calculate false negative t
			param[tc,0] = np.sum(matrix[tc,0:tc]) + np.sum(matrix[0:tc,after:until])

		param_micro = np.zeros(4)
		param_micro[3] = np.sum(param[:,3])
		param_micro[2] = np.sum(param[:,2])
		param_micro[1] = np.sum(param[:,1])
		param_micro[0] = np.sum(param[:,0])

		recall_micro = recall(param_micro)
		precision_micro = precision(param_micro)
		return (2 * precision_micro * recall_micro) / (precision_micro + recall_micro)

	elif select=='f1macro':
		matrix = confmatrix(t,p)
		target_n = len(np.unique(t))

		precisions = np.zeros(target_n)
		recalls = np.zeros(target_n)
		until = target_n - 1
		for tc in range(0, target_n):
			param = np.zeros(4)
			after = np.min((tc+1,target_n - 1))
			# calculate true positive t
			param[3] = matrix[tc,tc]
			# calculate true negative t
			param[2] = np.sum(matrix[0:tc,0:tc]) + np.sum(matrix[0:tc,after:until]) + np.sum(matrix[after:until,0:tc]) + np.sum(matrix[after:until,after:until])
			# calculate false positive t
			param[1] = np.sum(matrix[0:tc,tc]) + np.sum(matrix[after:until,tc])
			# calculate false negative t
			param[0] = np.sum(matrix[tc,0:tc]) + np.sum(matrix[tc,after:until])

			recalls[tc] = recall(param)
			precisions[tc] = precision(param)

		recall_macro = np.mean(recalls)
		precision_macro = np.mean(precisions)

		return (2 * precision_macro * recall_macro) / (precision_macro + recall_macro)

def train_nn(d, t, epoch=5000, maxmse=0.1, learning_r=0.1, reg_lambda=1, h_layers=[1], desclearning=False):
	randlimit = 2
	inputdim = len(d[0,:])
	ouputdim = len(t[0,:])
	h_layer_size = len(h_layers)
	
	w = [];	b = []; o = []
	wstar = []; bstar = []
	
	# init weight and bias between input and hidden layer
	w.append(np.random.rand(inputdim, h_layers[0]) * randlimit - (randlimit/2))
	b.append(np.random.rand(h_layers[0]) * randlimit - (randlimit/2))
	
	# init weight and bias between hidden layers
	for h in range(1, h_layer_size):
		w.append(np.random.rand( h_layers[h-1], h_layers[h]) * randlimit - (randlimit/2))
		b.append(np.random.rand(h_layers[h]) * randlimit - (randlimit/2))
	
	# init weight and bias between last hidden layer and output layer
	w.append(np.random.rand(h_layers[-1], ouputdim) * randlimit - (randlimit/2))
	b.append(np.random.rand(ouputdim) * randlimit - (randlimit/2))
	msebest = np.inf
	for ep in range(0, epoch):		
		if (ep + 1 % 200 == 0 and desclearning and learning_r >= 0.001): learning_r -= 0.0001
		D = []; B = []; A = []
		A.append(d)
		err = None
		# forward
		x = sigmo(d.dot(w[0]) + b[0])
		A.append(x)
		
		for h in range(1, h_layer_size):
			y = sigmo(x.dot(w[h]) + b[h])
			A.append(y)
			x = y
		
		y = sigmo(x.dot(w[-1]) + b[-1])
		
		# backward
		E = t - y
		mseepoch = np.mean(np.mean(E**2,axis=1))
		dw = E * sigmo(y, derive=True) * learning_r
		db = learning_r * np.sum(dw,axis=0)
		D.append(dw)
		B.append(db)
		
		for h in range(h_layer_size-1, -1,-1):
			err = dw.dot(w[h+1].T)	
	
			dw = err * sigmo(A[h+1], derive=True) * learning_r
			db = learning_r * np.sum(dw,axis=0)
			D.append(dw)
			B.append(db)

		for l in range(h_layer_size, -1,-1):

			w[l] += A[l].T.dot(D[h_layer_size - l]) * learning_r
			b[l] += B[h_layer_size - l]
			pass

		del A, D, B
		if msebest > mseepoch: 
			wstar = w; bstar = b
		print('epoch: ',ep,'\tmse:', mseepoch)
		if maxmse > mseepoch:
			break
	return w, b
	
def test_nn(d, t, w, b, h_layers=[1]):
	h_layer_size = len(h_layers)
	A = []
	A.append(d)
	err = None
	# forward
	x = sigmo(d.dot(w[0]) + b[0])
	A.append(x)
	
	for h in range(1, h_layer_size):
		y = sigmo(x.dot(w[h]) + b[h])
		A.append(y)
		x = y
	
	y = sigmo(x.dot(w[-1]) + b[-1])
	
	return y
def sigmo(x, derive=False):
	if(derive):
		return x*(1 - x)
	return 1 / (1 + np.exp(-x))
	
def tanha(x, derive=False):
	if(derive):
		return 1 - np.power(np.tanh(x), 2)
	return np.tanh(x)
	
def size(m):
	return (len(m[:,0]),len(m[0,:]))
	
