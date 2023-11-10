# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: MOHAMED ATHIL B
RegisterNumber: 212222230081
```
```PY
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

import chardet 
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Result output :
![280069902-0ec58261-8df8-4f45-9c6e-440464f071d0](https://github.com/Bmohamedathil/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119560261/89a33b0a-b8a3-4d95-90d7-2d8b46800ec4)

### data.head() :
![280070157-74d967ba-5f1f-4996-b11d-c61d06c92c3d](https://github.com/Bmohamedathil/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119560261/96b04874-a89d-4beb-8c3a-1f512b1a2221)


### data.info() :
![280071015-8267b319-1e42-4892-969d-2afeae06d4ce](https://github.com/Bmohamedathil/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119560261/01b2b58f-a6cd-4b29-93d0-653b47249ae2)

### data.isnull().sum() :
![280071339-34e8d223-7ad9-47b1-857b-1642f35c40b7](https://github.com/Bmohamedathil/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119560261/f158727e-44c5-4457-9a34-34017ba78838)

### Y_prediction value :
![280071554-72d1f9a9-31aa-449c-b46c-26219024ddca](https://github.com/Bmohamedathil/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119560261/826b1c68-a551-494c-96d4-fe08d5a05ba1)

### Accuracy value :
![280071681-8cde9bc5-164e-4de5-a126-a2aedda4ddc0](https://github.com/Bmohamedathil/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119560261/a75c6b9f-996d-429b-bb7c-02fcc67a87dd)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
