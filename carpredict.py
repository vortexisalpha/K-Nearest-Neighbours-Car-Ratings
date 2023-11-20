import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')


#need to convert our data into numerical data:

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))
#this automatically converts everything into neumerical 
# data and stores each collumn as variables

#makes this into a huge list
x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)

#this is the thing we want to predict
predict = 'class'

# x vs y split the data up into test and train data
x_train,x_test,y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

#create model and train it using the split data
model = KNeighborsClassifier(n_neighbors=9) #parameter is the number of neighbors u want k to be

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print(acc)

predicted = model.predict(x_test)

names = ['unacc', 'acc', 'good', 'vgood']
for i in range(len(x_test)):
    {
        print('Predicted: ',names[predicted[i]],  'Actual: ', names[y_test[i]], '\nData: ', x_test[i] )
    }