import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.

# def part1_scatter():
#     plt.figure()
#     plt.scatter(X_train, y_train, label='training data')
#     plt.scatter(X_test, y_test, label='test data')
#     plt.legend(loc=4);

#### ============================================================
####                        Question 1
#### ============================================================


def answer_one():
    from sklearn.linear_model import LinearRegression 
    from sklearn.preprocessing import PolynomialFeatures 
    # To capture interactions between the original features by adding them as features to the Linear model.

    clf                     = LinearRegression() 
    degree_predictions      = np.zeros((4,100)) 
    X_in                    = np.linspace(0,10,100) #Given requirement 
    degrees                 = [1,3,6,9] #Given requirement

    for i in range(len(degrees)):

        poly = PolynomialFeatures(degrees[i]) # Object to add polynomial features.

        #Add polynomial features to training data and input data:
        #Need to transpose X_train and X_input for poly fit to Mork.

        X_train_poly = poly.fit_transform(X_train[None].T) 
        X_input_poly = poly.fit_transform(X_in[None].T)

        #Train Linear regression classifier with training data:
        clf.fit(X_train_poly, y_train)

        #Get predictions from Linear classifier using transformed input data:
        degree_predictions[i,:] = clf.predict(X_input_poly)

    return degree_predictions

#### ============================================================
####                        Question 2
#### ============================================================

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score

    r2_train = np.array([])
    r2_test = np.array([])

    # Range of degrees
    degrees = range(10)
    
    for i in degrees :

        poly = PolynomialFeatures(i)
        X_train_poly    = poly.fit_transform(X_train[None].T)
        X_test_poly     = poly.fit_transform(X_test[None].T)

        linreg = LinearRegression().fit(X_train_poly, y_train)

        r2_train    = np.append(r2_train, linreg.score(X_train_poly, y_train))
        r2_test     = np.append(r2_test, linreg.score(X_test_poly, y_test))
    
    return (r2_train, r2_test)

print(answer_two())

#### ============================================================
####                        Question 3
#### ============================================================

def answer_three():

    r2 = answer_two()

    plt.figure(figsize=(10,5))
    plt.plot(range(10), r2[0], 'r', label='training data') 
    plt.plot(range(10), r2[1], 'b', label='test data') 
    plt.xlabel('Order') 
    plt.ylabel('Score') 
    plt.legend()
    plt.show()

    return (0,9,6)

answer_three()