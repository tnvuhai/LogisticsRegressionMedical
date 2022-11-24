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
corr = dataset.corr()
hm = sns.heatmap(corr, annot = True)
hm.set(xlabel='\nMedical Columns', ylabel='Medical columns\t', title = "Correlation matrix of data\n")
plt.savefig("Correlation matrix of data.png")

# input
x = dataset.iloc[:, [1,2,3,4,5,6,7,8]].values

# output
y = dataset.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(dataset[["gos","training2","creadit3","capital","gender","experience","age","education"]], dataset["qd"], test_size = 0.2, random_state = 0)
classifier = LogisticRegression(random_state = 6, fit_intercept = True)
classifier.fit(X_train, y_train)

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
    for i in range(0,8):
        stringC1 = FinalCondition[i][0]
        stringC2 = FinalCondition[i][1]
        Scondition = f"""
if(x{i}Coef {stringC1} 0 and x{i}ZValue {stringC2} 1.96):
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
for i in range(1,9):
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
    X_train, X_test, y_train, y_test = train_test_split(dataset[["gos","training2","creadit3","capital","gender","experience","age","education"]], dataset["qd"], test_size=0.05)
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit(disp=0)
    Check = 0
    for i in range(8):
        exec(f"x{i}ZValue = result.tvalues[{i}]")
    for i in range(8):
        exec(f"x{i}Coef = result.params[{i}]")
    CheckCondition()
    # If Check passed 8 tests
    if (Check == 8):
        print(result.summary())
        TargetDataFrameTrain = pd.DataFrame(X_train, columns=["gos", "training2", "creadit3", "capital", "gender", "experience",
                                                     "age", "education","STT"])
        TargetDataFrameTrain.insert(8, "qd", y_train, True)
        TargetDataFrameTrain.to_excel("Traindata.xlsx")

        TargetDataFrameTest = pd.DataFrame(X_test,
                                            columns=["gos", "training2", "creadit3", "capital", "gender", "experience",
                                                     "age", "education"])
        TargetDataFrameTest.insert(8, "qd", y_test, True)
        TargetDataFrameTest.to_excel("Testdata.xlsx")
        break
    runningTime += 1
    if(runningTime == runningLimit):
        print("No optimized solution found! End Loop")
        break

#Print model Evaluation
y_pred = classifier.predict(X_test)
qd = confusion_matrix(y_test, y_pred)
qd = pd.DataFrame(qd)
print("\nConfusion Matrix : \n", qd)
print(f"\nAccuracy of training: {classifier.score(X_train, y_train)}")
print(f"\nAccuracy of testing: {classifier.score(X_test, y_test)}")
