import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

#dataset
train_X= idx2numpy.convert_from_file('data/train-images-idx3-ubyte')
train_Y= idx2numpy.convert_from_file('data/train-labels-idx1-ubyte')
test_X= idx2numpy.convert_from_file('data/t10k-images-idx3-ubyte')
test_Y= idx2numpy.convert_from_file('data/t10k-labels-idx1-ubyte')
#processed original data sets
test_Y_org=test_Y.reshape(test_Y.shape[0],1)
test_set_X = test_X.reshape(test_X.shape[0],-1).T
#datasets for testing
train_set_Y=train_Y.reshape(train_Y.shape[0],1)
test_set_Y=train_Y.reshape(train_Y.shape[0],1)

test_set_x_flatten = train_X.reshape(train_X.shape[0],-1).T
train_set_x_flatten = train_X.reshape(train_X.shape[0],-1).T
def onehotit(z):
    s=np.zeros((z.shape[0],int(np.amax(z)+1)))
    for i in range(z.shape[0]):
        s[i,z[i][0]]=1
    return s

def softmax(A):
    A-=np.max(A)
    A_exp=np.exp(A)
    A_sum=(np.sum(A_exp,axis=0)).reshape((1,A.shape[1]))
    s=A_exp/A_sum
    return s

def costp(A,X,Y,w):
    cost_i=softmax(A)
    cost=np.sum(Y*np.log(cost_i).T)/Y.shape[0] +np.sum(w*w)*0.5
    grad=(-1/Y.shape[0])*(np.dot(X,(Y-cost_i.T))) + w
    return cost,grad

def prob_and_pred(X,w):
    prob=softmax(np.dot(w.T,X))
    pred=(np.argmax(prob,axis=0)).reshape((1,X.shape[1]))
    return pred,prob



learning_rate=1e-5
num_iteration=1000
train_Y1=onehotit(train_set_Y)
w=np.zeros((test_set_x_flatten.shape[0],10))
costs=[]
def accuracy(X,Y):
    s=np.sum(X.T==Y)/X.shape[1]
    return s*100

def optimize(w,X,Y,learning_rate,num_iteration,costs):

    for i in range(num_iteration):
        cost,grad=costp(np.dot(w.T,X),X,Y,w)
        if i%50==0:
            costs.append(-cost)
            print(-cost)
        w=w-learning_rate*grad

    plt.plot(costs)
    plt.show()
    return w,costs,grad,cost,i

w,costs,grad,cost,i=optimize(w,train_set_x_flatten,train_Y1,learning_rate,num_iteration,costs)
pred1,prob1=prob_and_pred(train_set_x_flatten,w)
pred2,prob2=prob_and_pred(test_set_x_flatten,w)
print(accuracy(pred1,train_set_Y))
print(accuracy(pred2,test_set_Y))
    # 92% accuracy
