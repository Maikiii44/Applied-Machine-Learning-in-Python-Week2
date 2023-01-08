import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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

print(answer_one())