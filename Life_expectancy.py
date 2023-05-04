import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Life_expectancy.csv')
print(data.head())
plt.bar(data['Life expectancy(M)'],data['Life expectancy(F)'])
plt.show()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data[['Life expectancy(M)']],data[['Life expectancy(F)']])
print(model.predict([[80]]))