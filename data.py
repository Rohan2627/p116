import pandas as pd
import plotly_express as py
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("iPhone.csv")

salary = df["EstimatedSalary"].tolist()
purchase = df["Purchased"].tolist()
# age = data["Pur"].tolist()



fig = py.scatter(x = salary, y = purchase)

fig.show()  

# -------------- figuring out the planets avlb o groudw


import plotly.graph_objects as go

salaries = df["EstimatedSalary"].tolist()
ages = df["Age"].tolist()

purchased = df["Purchased"].tolist()
colors=[]
for data in purchased:
  if data == 1:
    colors.append("green")
  else:
    colors.append("red")



fig = go.Figure(data=go.Scatter(
    x=salaries,
    y=ages,
    mode='markers',
    marker=dict(color=colors)
))
fig.show()

# ---------------------------------------------------------

from sklearn.model_selection import train_test_split 

salary_train, salary_test, purchase_train, purchase_test = train_test_split(factors, purchases, test_size = 0.25, random_state = 0)




print(salary_train[0:10])

from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 

salary_train = sc_x.fit_transform(salary_train)  
salary_test = sc_x.transform(salary_test) 
  
print (salary_train[0:10])

from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(salary_train, purchase_train)





 

pd.option_con