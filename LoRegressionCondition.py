import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
import math
import warnings
warnings.filterwarnings("ignore")

#Read Data and show the corelation matrix of Data
dataset = pd.read_excel("DataSet.xlsx")
q_low = dataset["age"].quantile(0.05)
q_hi  = dataset["age"].quantile(0.95)
dataset = dataset[(dataset["age"] < q_hi) & (dataset["age"] > q_low)]

corr = dataset.corr()
hm = sns.heatmap(corr, annot = True)
hm.set(xlabel='\nMedical Columns', ylabel='Medical columns\t', title = "Correlation matrix of data\n")
plt.savefig("Correlation matrix of data.png")

# Change the columns as followed list
ColumnsName = ["gos","training2","creadit3","capital","gender","experience","age","education"]
LenOfColumnList = len(ColumnsName)

X_train, X_test, y_train, y_test = train_test_split(sm.add_constant(dataset[ColumnsName]), dataset["qd"], test_size = 0.05, random_state = 0)


def ConvertCondition(string):
    string = string.lower()
    string = string.replace("+",">")
    string = string.replace("yes",">")
    string = string.replace("no","<")
    string = string.replace("-","<")
    Condition = string.split(", ")
    return Condition

Check = 0
def CheckCondition():
    for i,j in zip(range(LenOfColumnList),range(1,LenOfColumnList+1)):
        stringC1 = FinalCondition[i][0]
        stringC2 = FinalCondition[i][1]
        Scondition = f"""
if(x{j}Coef {stringC1} 0 and x{j}ZValue {stringC2} 1.96):
    Check += 1
        """
        exec(Scondition, globals())

#Start input condition
IntroString = """
Note:
Please input in right structured format:
+: for positive coef
-: for negative coef
yes: for > 1.96 Z value
no: for < 1.96 Z value
Example: if you want x1 is positve and > 1.96 value:
input this: +, yes
(please input a space after comma like example, and do not this: "+,no") 
"""
print(IntroString)
FinalCondition = []
for i in range(1,LenOfColumnList+1):
    exec(f"""
while(True):
    Condition = ConvertCondition(input("Input condition of x{i}:")) 
    if(len(Condition) < 2):
        Condition = ConvertCondition(input("Input condition of x{i}:")) 
    if(len(Condition) > 1):
        FinalCondition.append(Condition)
        break
    """)

#Train model and print result
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit(disp=0)

runningLimit = int(input("Please input running times for searching the expected result:"))
runningTime = 0
while(True):
    X_train, X_test, y_train, y_test = train_test_split(sm.add_constant(dataset[ColumnsName]), dataset["qd"], test_size=0.05)
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit(disp=0)
    Check = 0
    for i in range(1, LenOfColumnList+1):
        exec(f"x{i}ZValue = result.tvalues[{i}]")
    for i in range(1, LenOfColumnList+1):
        exec(f"x{i}Coef = result.params[{i}]")
    CheckCondition()
    # If Check passed all tests
    if (Check == LenOfColumnList):
        print(result.summary())
        TargetDataFrameTrain = pd.DataFrame(X_train, columns=ColumnsName)
        TargetDataFrameTrain.insert(LenOfColumnList, "qd", y_train, True)
        TargetDataFrameTrain.to_excel("Traindata.xlsx")

        TargetDataFrameTest = pd.DataFrame(X_test,
                                            columns=ColumnsName)
        TargetDataFrameTest.insert(LenOfColumnList, "qd", y_test, True)
        TargetDataFrameTest.to_excel("Testdata.xlsx")
        break
    runningTime += 1
    if(runningTime == runningLimit):
        print("No optimized solution found! End Loop")
        break

#Print model Evaluation
y_pred = result.predict(X_test)
y_pred = y_pred.apply(lambda x: 1 if x > 0.5 else 0)
qd = confusion_matrix(y_test, y_pred)
qd = pd.DataFrame(qd)
print("\nConfusion Matrix : \n", qd)

