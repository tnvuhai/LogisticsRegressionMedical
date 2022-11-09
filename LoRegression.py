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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
classifier = LogisticRegression(random_state = 6, fit_intercept = True)
classifier.fit(X_train, y_train)

#Train model and print result
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit(disp=0)
# display(result.summary())

while (True):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit(disp=0)

    x3ZValue = result.tvalues[2]
    x5ZValue = result.tvalues[4]

    x3Coef = result.params[2]
    x5Coef = result.params[4]
    if ((abs(x3ZValue) > 1.96 and abs(x5ZValue) < 1.96) and (x3Coef > 0 and x5Coef > 0)):
        IntroString = """
            Run LOOP till satistfied the condition:
                x3 > 0 and |z| > 1.96
                x5 > 0 and |z| < 1.96
        """
        print(IntroString)
        print(result.summary())
        break

#Print model Evaluation
y_pred = classifier.predict(X_test)
qd = confusion_matrix(y_test, y_pred)
qd = pd.DataFrame(qd)
print("\nConfusion Matrix : \n", qd)
print(f"\nAccuracy of training: {classifier.score(X_train, y_train)}")
print(f"\nAccuracy of testing: {classifier.score(X_test, y_test)}")

#Test with real data
def PredictPro(gos, training, credit, experience, intercept):
    gosPara = classifier.coef_[0][0]
    trainingPara = classifier.coef_[0][1]
    creditPara = classifier.coef_[0][2]
    experiencePara = classifier.coef_[0][5]
    logarit = intercept + gosPara * gos - trainingPara * training + \
              creditPara * credit + experiencePara * experience
    Decision = 1 / (1 + math.exp(-logarit))

    if (Decision >= 0.5):
        return 1
    else:
        return 0

#You can change the parameter inside:
"""
Biến phụ thuộc: qd (Quyết định nghỉ việc)
Biến độc lập: Đã giải trình trong excel

qd = gos+ training2+ creadit3 + capital + gender+ experience + age + education

condition:

gos :        +
training 2:  +
creadit 3:   -
experience:  +
"""

print("\nTest with real data\nGos = 1, training = 0, credit = 1, experience = 8")
QdDecision = PredictPro(1,0,1,8, classifier.intercept_)

if(QdDecision == 1):
    print("-->Qd = 1")
else:
    print("-->Qd = 0")
