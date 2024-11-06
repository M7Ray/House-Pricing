# House-Pricing

- PROJECT OVERVIEW

    The main goal of this project, as a data analyst for a Real Estate Investment Trust (REIT) looking to enter the residential real estate market, is to ascertain the         market value of houses based on a variety of property attributes. In this project, a dataset from that includes information on homes sold between May 2014 and May 2015     is analyzed. Key features are used to explore, analyze, and create predictive models for housing prices.

    The data set contains the following parameters:
    Basic Information: id, date, price
    Property Characteristics: bedrooms, bathrooms, sqft_living, sqft_lot
    Structural Details: floors, waterfront, view, condition, grade
    Location Information: zipcode, lat, long
    Recent Living and Lot Size (2015): sqft_living15, sqft_lot15
    also: sqft_above, sqft_basement, yr_built, yr_renovated

- INSIGHTS

    1- Impact of Living Space and Lot Size: Properties with larger sqft_living and sqft_lot generally exhibit higher prices, as expected. The relationship is especially           strong in premium locations, such as waterfront areas.
  
    2- Renovations and Recent Updates: Houses with recent renovations or updates (reflected by yr_renovated, sqft_living15, and sqft_lot15) tend to have higher prices, 
       possibly due to improved structural and aesthetic conditions.
  
    3- Location and Zip Code: Properties in certain zip codes show significant variations in price, emphasizing the importance of location in real estate pricing.
  
    4- House Grade and Condition: Houses with higher grades and better conditions command higher prices, as these factors directly affect the perceived value and quality 
       of the home.

- CONCLUSION

    The project illustrates how to analyze and comprehend different property features in order to create a predictive model for house values. The findings give the Real 
    Estate Investment Trust important information on the factors that have the most effects on home values. The REIT can use these results to guide its investment choices,     concentrating on properties with attributes that greatly raise their market worth.

The dataset used in this project is available on Kaggle and can be accessed [here](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/code)


```python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
%matplotlib inline


#importing dataset
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

<img width="125" alt="types" src="https://github.com/user-attachments/assets/cc390372-9e91-42cc-92de-72ae1ee8b894">
<img width="528" alt="describe" src="https://github.com/user-attachments/assets/6acecb73-b143-4a11-b6ec-b146933be249">


Data Wrangling - Data Cleaning


    
```python

df.drop(columns=['id', 'Unnamed: 0'], inplace=True)
print(df.describe())

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

###    output:
## number of NaN values for the column bedrooms : 13
## number of NaN values for the column bathrooms : 10

#replace missing values:
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
df['bathrooms'].replace(np.nan,mean, inplace=True)

#check again for number of missing values
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

###    output
## number of NaN values for the column bedrooms : 0
## number of NaN values for the column bathrooms : 0


```

Exploratory Data Analysis (Distribution Analysis, Outlier Detection, Correlation Analysis)


```python
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

<img width="58" alt="floors" src="https://github.com/user-attachments/assets/c5114e2c-6d16-4dc8-9615-6602a3e107d8">
<img width="136" alt="type2" src="https://github.com/user-attachments/assets/6e0fdc84-0e05-44ca-90d7-8d50fe42a7ce">
<img width="522" alt="plot" src="https://github.com/user-attachments/assets/c4abb2c1-12a2-43ef-b5c9-87a5b8b3bb26">
<img width="518" alt="plot2" src="https://github.com/user-attachments/assets/9dce24a5-6136-490c-b624-ea1a3dd0b66d">


Model Development (Linear Regression, Pipeline and Polynomial Regression)



```python

# linear regression model and R^2
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)
## output: 0.00046769430149007363

#linear regression model to predict the 'price' using the feature 'sqft_living' and R^2
X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
print(lm.score(X,Y))
## output: 0.4928532179037931

# linear regression model to predict the 'price' using the list of features, find R^2
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
Z = df[["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]]
Y = df['price']
lm.fit(Z, Y)
r_squared = lm.score(Z, Y)
print(lm.score(Z,Y))
## output: 0.6576890354915759


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
## output: R^2 Score: 0.7512051345272872


```

Model Evaluation (Train-Test, Polynomial Features with Ridge Regression)



```python

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
## output: number of test samples: 3242
##         number of training samples: 18371

# Fitting a Ridge regression object using the training data with parameter 0.1, find R^2 using test data
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
RigeModel = Ridge(alpha=0.1)
RigeModel.fit(x_train, y_train)
y_pred = RigeModel.predict(x_test)
r_squared = r2_score(y_test, y_pred)
print("R^2 Score:", r_squared)
## output: R^2 Score: 0.647875916393907

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
## output: R^2 Score: 0.7002744263583341



```




