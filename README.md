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

#Data Wrangling

```ruby



print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

#replace missing values:

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
df['bathrooms'].replace(np.nan,mean, inplace=True)

```


















