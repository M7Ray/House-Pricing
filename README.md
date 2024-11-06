# House-Pricing

Using online jupyter notebook
```ruby


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline

##
#importing dataset
##


import piplite
await piplite.install('seaborn')


from pyodide.http import pyfetch
async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
await download(filepath, "housing.csv")
file_name="housing.csv"



# Load the csv:
df = pd.read_csv(file_name)

# display first 5 columns and the datatypes:
df.head()
print(df.dtypes)


# Statistical summary
df.describe()

```       


Data Wrangling

```ruby

df.drop(columns=['id', 'Unnamed: 0'], inplace=True)
print(df.describe())

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


#replace missing values:
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
df['bathrooms'].replace(np.nan,mean, inplace=True)

#check again for number of missing values
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

```

Exploratory Data Analysis

```ruby
#unique values:
df['floors'].value_counts().to_frame()

#determine whether houses with a waterfront view or without a waterfront view have more price outliers.
plt.figure(figsize=(10, 6))
sns.boxplot(x='waterfront', y='price', data=df)
plt.xlabel('Waterfront View')
plt.ylabel('Price')
plt.title('Price Distribution: Waterfront View vs No Waterfront View')
plt.show()


#determine if the feature sqft_above is negatively or positively correlated with price
plt.figure(figsize=(10, 6))
sns.regplot(x='sqft_above', y='price', data=df)
plt.show()

#find the feature other than price that is most correlated with price.
df.corr()['price'].sort_values()

```

Model Development


```ruby

# linear regression model and R^2
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)

#linear regression model to predict the 'price' using the feature 'sqft_living' and R^2
X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
print(lm.score(X,Y))

# linear regression model to predict the 'price' using the list of features, find R^2
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
Z = df[["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]]
Y = df['price']
lm.fit(Z, Y)
r_squared = lm.score(Z, Y)
print(lm.score(Z,Y))


# creating list of tuples where the first element in the tuple contains the name of the estimator (scale, polynomial, model) and the
# second element in the tuple contains the model constructor (StandardScaler, PolynomialFeatures, LinearRegression)
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


#create a pipeline object to predict the 'price', fit the object, find R^2

X = df[features]
Y = df['price']

pipe = Pipeline(Input)
pipe.fit(X, Y)
r_squared = pipe.score(X, Y)
print("R^2 Score:", r_squared)

```

Model Evaluation

```ruby

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")

#  split the data into training and testing sets
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

# Fitting a Ridge regression object using the training data with parameter 0.1, find R^2 using test data
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
RigeModel = Ridge(alpha=0.1)
RigeModel.fit(x_train, y_train)
y_pred = RigeModel.predict(x_test)
r_squared = r2_score(y_test, y_pred)
print("R^2 Score:", r_squared)

# second order polynomial transform on both the training data and testing data
# fitting a Ridge regression object using the training data with parameter 0.1, find R^2
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
RidgeModel_poly = Ridge(alpha=0.1)
RidgeModel_poly.fit(x_train_poly, y_train)
y_pred_poly = RidgeModel_poly.predict(x_test_poly)
r_squared_poly = r2_score(y_test, y_pred_poly)
print("R^2 Score:", r_squared_poly)

```




