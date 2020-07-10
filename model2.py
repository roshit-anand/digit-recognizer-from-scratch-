import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
#dataset
train_X= idx2numpy.convert_from_file('data/mnist/train-images-idx3-ubyte')
train_Y= idx2numpy.convert_from_file('data/mnist/train-labels-idx1-ubyte')
test_X= idx2numpy.convert_from_file('data/mnist/t10k-images-idx3-ubyte')
test_Y= idx2numpy.convert_from_file('data/mnist/t10k-labels-idx1-ubyte')
#processed original data sets
test_Y_org=test_Y.reshape(test_Y.shape[0],1)
test_set_X = test_X.reshape(test_X.shape[0],-1).T
#datasets for testing
train_set_Y=train_Y.reshape(train_Y.shape[0],1)
test_set_Y=test_Y.reshape(test_Y.shape[0],1)

test_set_x_flatten = test_X.reshape(test_X.shape[0],-1).T
train_set_x_flatten = train_X.reshape(train_X.shape[0],-1).T
def relu(x):
    s=x*(x>0)
    return s

def drelu(x):
    s=1*(x>0)
    return s

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

def costp(A,X,Y,w,b):
    cost_i=softmax(A)
    cost=np.sum(Y*np.log(cost_i).T)/Y.shape[0]
    grad_w=(-1/Y.shape[0])*(np.dot(X,(Y-cost_i.T)))
    grad_A=(-1/Y.shape[0])*(np.dot(w,(Y-cost_i.T).T))
    grad_b=(-1/Y.shape[0])*(np.dot(np.ones((1,60000)),Y-cost_i.T)).T
    return cost,grad_w,grad_A,grad_b

def prob_and_pred(X,w1,w2,w3):
    prob2=relu(np.dot(w1.T,X))
    prob1=relu(np.dot(w2.T,prob2))
    prob=softmax(np.dot(w3.T,prob1))
    pred=(np.argmax(prob,axis=0)).reshape((1,X.shape[1]))
    return pred,prob

def accuracy(X,Y):
    s=np.sum(X.T==Y)/X.shape[1]
    return s*100

#layers  784 > 128 > 10

def optimize(w1,w2,w3,b1,b2,b3,beta1,beta2,X,Y,learning_rate,num_iteration,costs):
    moment1_w1=0
    moment1_w2=0
    moment1_w3=0
    moment1_b1=0
    moment1_b2=0
    moment1_b3=0
    moment2_w1=0
    moment2_w2=0
    moment2_w3=0
    moment2_b1=0
    moment2_b2=0
    moment2_b3=0

    for i in range(1,num_iteration):
        X1=np.dot(w1.T,X)+b1
        A1=relu(X1)
        X2=np.dot(w2.T,A1)+b2
        A2=relu(X2)
        cost,grad_w3,grad_A,grad_b3=costp(np.dot(w3.T,A2),A2,Y,w3,b3)
        if i%1==0:
            costs.append(-cost)
            print(-cost,"  ",i)
        grad_w1=np.dot(X,(np.dot(w2,grad_A*drelu(A2))*drelu(A1)).T)
        grad_w2=np.dot(A1,(drelu(A2)*grad_A).T)
        grad_b1=np.dot(np.ones((1,60000)),(np.dot(w2,grad_A*drelu(A2))*drelu(A1)).T).T
        grad_b2=np.dot(np.ones((1,60000)),(grad_A*drelu(A2)).T).T
                        #adam
        moment1_w1=beta1*moment1_w1 + (1-beta1)*grad_w1
        moment1_w2=beta1*moment1_w2 + (1-beta1)*grad_w2
        moment1_w3=beta1*moment1_w3 + (1-beta1)*grad_w3
        moment1_b1=beta1*moment1_b1 + (1-beta1)*grad_b1
        moment1_b2=beta1*moment1_b2 + (1-beta1)*grad_b2
        moment1_b3=beta1*moment1_b3 + (1-beta1)*grad_b3

        moment2_w1=beta2*moment2_w1 + (1-beta2)*(grad_w1**2)
        moment2_w2=beta2*moment2_w2 + (1-beta2)*(grad_w2**2)
        moment2_w3=beta2*moment2_w3 + (1-beta2)*(grad_w3**2)
        moment2_b1=beta2*moment2_b1 + (1-beta2)*(grad_b1**2)
        moment2_b2=beta2*moment2_b2 + (1-beta2)*(grad_b2**2)
        moment2_b3=beta2*moment2_b3 + (1-beta2)*(grad_b3**2)

        unbias1_w1=moment1_w1/(1-beta1**i)
        unbias1_w2=moment1_w2/(1-beta1**i)
        unbias1_w3=moment1_w3/(1-beta1**i)
        unbias1_b1=moment1_b1/(1-beta1**i)
        unbias1_b2=moment1_b2/(1-beta1**i)
        unbias1_b3=moment1_b3/(1-beta1**i)

        unbias2_w1=moment2_w1/(1-beta2**i)
        unbias2_w2=moment2_w2/(1-beta2**i)
        unbias2_w3=moment2_w3/(1-beta2**i)
        unbias2_b1=moment2_b1/(1-beta2**i)
        unbias2_b2=moment2_b2/(1-beta2**i)
        unbias2_b3=moment2_b3/(1-beta2**i)

        w1=w1-learning_rate*(unbias1_w1/(np.sqrt(unbias2_w1)+1e-7))
        w2=w2-learning_rate*(unbias1_w2/(np.sqrt(unbias2_w2)+1e-7))
        w3=w3-learning_rate*(unbias1_w3/(np.sqrt(unbias2_w3)+1e-7))

        b1=b1-learning_rate*(unbias1_b1/(np.sqrt(unbias2_b1)+1e-7))
        b2=b2-learning_rate*(unbias1_b2/(np.sqrt(unbias2_b2)+1e-7))
        b3=b3-learning_rate*(unbias1_b3/(np.sqrt(unbias2_b3)+1e-7))

    plt.plot(costs)
    plt.show()
    return w1,w2,w3,b1,b2,b3,costs,grad_A,cost,i

learning_rate=1e-3
num_iteration=1000
train_Y1=onehotit(train_set_Y)
w1=np.random.rand(test_set_x_flatten.shape[0],512)*0.0001
w2=np.random.rand(512,128)*0.0001
w3=np.random.rand(128,10)*0.0001
b1=np.random.rand(512,1)*0.0001
b2=np.random.rand(128,1)*0.0001
b3=np.random.rand(10,1)*0.0001
beta1=0.9
beta2=0.99
costs=[]

w1,w2,w3,b1,b2,b3,costs,grad_A,cost,i=optimize(w1,w2,w3,b1,b2,b3,beta1,beta2,train_set_x_flatten,train_Y1,learning_rate,num_iteration,costs)
pred1,prob1=prob_and_pred(train_set_x_flatten,w1,w2,w3)
pred2,prob2=prob_and_pred(test_set_x_flatten,w1,w2,w3)
print(accuracy(pred1,train_set_Y))
print(accuracy(pred2,test_set_Y))
