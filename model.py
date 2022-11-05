import pandas
from sklearn import linear_model
import pickle
df = pandas.read_csv("data.csv")

X = df[['Wieght', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
model = regr.fit(X.values, y)
filename = 'finalized_model.pkl'
pickle.dump(model, open(filename, 'wb'))