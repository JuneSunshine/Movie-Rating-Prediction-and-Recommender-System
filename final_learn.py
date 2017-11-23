'''
CSCI 630 Final Project: Movie IMDB Score Prediction

@author: Jingyang Li

@instructor: Yuxiao Huang
'''
import numpy as np
from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import  AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def setup_data_and_label(df):
    '''
    prepare features and label
    :param df: input dataframe
    :return: features and label
    '''
    num_instances = df.shape[0]
    # X - features, y - label
    # mixed weight for linear regression
    X1 = np.array(df['mixed_weight'],dtype='float64').reshape((num_instances,1))
    # three seperate features for the rest of methods
    X2 = np.array([df['plot_keywords_weight'],df['director_weight'],df['actor_weight']], dtype='float64').reshape((num_instances,3))
    # label - imdb score
    y = np.array(df['imdb_score'], dtype='float64').reshape((num_instances,1))

    return X1, X2, y

def linear(data):
    '''
    linear regression
    :param data: input data
    :return: None
    '''
    # get features and label
    X = setup_data_and_label(data)[0]
    y = setup_data_and_label(data)[2]
    # scale X before fit the data
    X = preprocessing.scale(X)

    # split our data
    # 20% of the data will be used for testing
    num_instances = data.shape[0]
    num_train = round(0.8 * num_instances)
    X_train = X[:num_train]
    X_test = X[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    clf = LinearRegression()

    # fit the training data (already processed and scaled) to our model
    clf.fit(X_train, y_train)

    # accuracy
    accuracy = clf.score(X_test, y_test)

    print ("Test accuracy is ", accuracy)

    print ('Coefficients : ' , clf.coef_)

    print("Mean squared error: %.2f"
          % np.mean((clf.predict(X_test) - y_test) ** 2))

    print('Variance score: %.2f' % clf.score(X_test, y_test))

    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, clf.predict(X_test), color='blue',
             linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


def SGD(data):
    '''
    Stochastic Gradient Descent
    :param data: input data
    :return: None
    '''
    sgd = SGDClassifier(loss="hinge", penalty="l2")

    X, y = setup_data_and_label(data)[1:]

    # scale X before fit the data
    X = preprocessing.scale(X)

    # split our data
    # 20% of the data will be used for testing
    num_instances = data.shape[0]
    num_train = round(0.8 * num_instances)
    X_train = X[:num_train]
    X_test = X[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    sgd.fit(X_train,y_train.ravel().astype(int))

    print ("The accuracy is ", sgd.score(X_test,y_test.ravel().astype(int)))

def dctree(data):
    '''
    Multi-class Adaboosted Decision Tree
    :param data: input data
    :return: None
    '''
    dc = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=600,learning_rate=1.5,algorithm='SAMME')

    X, y = setup_data_and_label(data)[1:]

    # scale X before fit the data
    X = preprocessing.scale(X)

    # split our data
    # 20% of the data will be used for testing
    num_instances = data.shape[0]
    num_train = round(0.8 * num_instances)
    X_train = X[:num_train]
    X_test = X[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    dc.fit(X_train, y_train.ravel().astype(int))
    print ("The accuracy is ", dc.score(X_test,y_test.ravel().astype(int)))



def logistic(data):
    '''
    Multinomial Logistic Regression
    :param data: input data
    :return: None
    '''
    X, y = setup_data_and_label(data)[1:]

    # scale X before fit the data
    X = preprocessing.scale(X)

    # split our data
    # 20% of the data will be used for testing
    num_instances = data.shape[0]
    num_train = round(0.8 * num_instances)
    X_train = X[:num_train]
    X_test = X[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    clf = LogisticRegression(solver='sag',multi_class='multinomial').fit(X_train,y_train.ravel().astype(int))
    print("testing score : %.3f (%s)" % (clf.score(X_test, y_test.ravel().astype(int)), 'multinomial'))




