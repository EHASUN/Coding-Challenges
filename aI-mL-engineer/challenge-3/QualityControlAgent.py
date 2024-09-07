#Importing Required Libraries
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image  
import pydotplus
from six import StringIO 
from IPython.display import Image


#Loading Data
col_names = ['length', 'width', 'sleeve_length', 'defect_rate', 'batch_size']
# load dataset
pima = pd.read_csv("batch_data.csv")
#pima['NumericData_Float'] = pima['NumericData'].astype(float)


pima.head()
print(pima.head())



#Feature Selection
#Here, need to divide given columns into two types of variables dependent(or target variable) and independent variable(or feature variables).
#split dataset in features and target variable
feature_cols = ['length', 'width', 'sleeve_length', 'defect_rate', 'batch_size']
X = pima[feature_cols] # Features
y = pima.batch_size # Target variable

#Splitting Data
#Let's split the dataset by using the function train_test_split(). need to pass three parameters features; target, and test_set size.
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#Building Decision Tree Model
#Let's create a decision tree model using Scikit-learn.

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


#Evaluating the Model
#Accuracy can be computed by comparing actual test set values and predicted values.

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Visualizing Decision Trees
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('qualityControl.png')
Image(graph.create_png())

#Optimizing Decision Tree Performance
#criterion : optional (default=”gini”) or Choose attribute selection measure. This parameter allows us to use the different-different attribute selection measure. Supported criteria are “gini” for the Gini index and “entropy” for the information gain.

#splitter : string, optional (default=”best”) or Split Strategy. This parameter allows us to choose the split strategy. Supported strategies are “best” to choose the best split and “random” to choose the best random split.

#max_depth : int or None, optional (default=None) or Maximum Depth of a Tree. The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples. The higher value of maximum depth causes overfitting, and a lower value causes underfitting (Source).

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Visualizing Decision Trees
#Let's make our decision tree a little easier to understand using the following code: 

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('qualityControl.png')
Image(graph.create_png())

#Here, I've completed the following steps: 

#Imported the required libraries.
#Created a StringIO object called dot_data to hold the text representation of the decision tree.
#Exported the decision tree to the dot format using the export_graphviz function and write the output to the dot_data buffer.
#Created a pydotplus graph object from the dot format representation of the decision tree stored in the dot_data buffer.
#Written the generated graph to a PNG file named "diabetes.png".
#Displayed the generated PNG image of the decision tree using the Image object from the IPython.display module.






