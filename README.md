# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural network regression is a supervised learning method, and therefore requires a tagged dataset, which includes a label column. Because a regression model predicts a numerical value, the label column must be a numerical data type. You can train the model by providing the model and the tagged dataset as an input to Train Model.

In this experiment we need to develop a Neural Network Regression Model so first we need to create a dataset (i.e : an excel file with some inputs as well as corresponding outputs).Then upload the sheet to drive then using corresponding code open the sheet and then import the required python libraries for porcessing.

Use df.head to get the first 5 values from the dataset or sheet.Then assign x and y values for input and coressponding outputs.Then split the dataset into testing and training,fit the training set and for the model use the "relu" activation function for the hidden layers of the neural network (here two hidden layers of 5 and 10 neurons are taken to process).To check the loss mean square error is uesd.Then the testing set is taken and fitted, at last the model is checked for accuracy via preiction.

## Neural Network Model

![image](https://github.com/Vivekreddy8360/basic-nn-model/assets/94525701/b54d3977-22f9-4660-a836-fda7835eb016)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Developed by : M.vivek reddy
Reg : 212221240030
```
from google.colab import auth
auth.authenticate_user()
import gspread
from google.auth import default
creds,_=default()
gs=gspread.authorize(creds)
import pandas as pd
ws=gs.open('My sheet').sheet1
data1=ws.get_all_values()
df=pd.DataFrame(data1[1:],columns=data1[0])
df=df.astype({'INPUT':'float'})
df=df.astype({'OUTPUT':'float'})
df.head()
X=df[['INPUT']].values
y=df[['OUTPUT']].values
print(X)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Activation, Dense
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
model = Sequential([
    Dense(5,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss = 'mse')
model.fit(X_train1,y_train,epochs=2000)
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
model.evaluate(X_test1,y_test)
X_n1 = [[25]]
X_n1_1 = Scaler.transform(X_n1)
model.predict(X_n1_1)
## Dataset Information

![nn2](https://github.com/Vivekreddy8360/basic-nn-model/assets/94525701/3a11984d-44ee-4515-bdba-5174a1774ca3)


## OUTPUT


### Training Loss Vs Iteration Plot


![nn3](https://github.com/Vivekreddy8360/basic-nn-model/assets/94525701/f918d52f-a8e1-4822-b57c-f13fd49b3fe3)

### Test Data Root Mean Squared Error

![image](https://github.com/Vivekreddy8360/basic-nn-model/assets/94525701/ce84c0c5-5485-43c2-8526-4e5957bc92c3)


### New Sample Data Prediction

![nn5](https://github.com/Vivekreddy8360/basic-nn-model/assets/94525701/8209d18e-c38a-4633-bc6f-c6360c62d3aa)


## RESULT
