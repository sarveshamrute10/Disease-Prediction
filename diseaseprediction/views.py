from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request,"home.html")
def diabetes(request):
    return render(request,"diabetes.html")
def heart(request):
    return render(request,"heart.html")
def result1(request):
    df = pd.read_csv("S:\\SD\\Diabetes\\diabetes.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    val1 = float(request.GET['d1'])
    val2 = float(request.GET['d2'])
    val3 = float(request.GET['d3'])
    val4 = float(request.GET['d4'])
    val5 = float(request.GET['d5'])
    val6 = float(request.GET['d6'])
    val7 = float(request.GET['d7'])
    val8 = float(request.GET['d8'])
    pred = model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])

    result5 = ""
    if pred == [1]:
        result5 = "Positive"
    else:
        result5 = "Negative"
    return render(request,"diabetes.html",{"result2":result5})

def result3(request):
    df = pd.read_csv("S:\\SD\\Heart Disease\\heart.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)
    model = LogisticRegression(solver='liblinear', fit_intercept=False)
    model.fit(X_train, y_train)

    val9 = float(request.GET['h1'])
    val10 = float(request.GET['h2'])
    val11 = float(request.GET['h3'])
    val12 = float(request.GET['h4'])
    val13 = float(request.GET['h5'])
    val14 = float(request.GET['h6'])
    val15 = float(request.GET['h7'])
    val16 = float(request.GET['h8'])
    val17 = float(request.GET['h9'])
    val18 = float(request.GET['h10'])
    val19 = float(request.GET['h11'])
    val20 = float(request.GET['h12'])
    val21 = float(request.GET['h13'])
    pred1 = model.predict([[val9,val10,val11,val12,val13,val14,val15,val16,val17,val18,val19,val20,val21]])

    result6 = ""
    if pred1 == [1]:
        result6 = "Positive"
    else:
        result6 = "Negative"
    return render(request,"heart.html",{"result4":result6})