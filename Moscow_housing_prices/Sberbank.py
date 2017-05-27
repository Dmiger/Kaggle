import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls"]).decode("utf8"))

df = pd.read_csv("train.csv")
print(df.columns)

#Group by data type
df.columns.to_series().groupby(df.dtypes).groups

#List column titles
list(df.select_dtypes(include=['int64']).columns)

round(df.select_dtypes(include=['int64']).describe(),2)

#Number of rows with missing values
print(df.shape[0] - df.dropna().shape[0])

print(df['num_room'].describe())

#Missing values
x = df.isnull().sum()
x_ascend = x.sort_values(ascending = [False])
x_asc_df = pd.DataFrame(x_ascend)
#Number of rows without missing values.
#51 variables from 292 contain missing values.
np.count_nonzero(x_asc_df)

#Which of them stand out? I slice the first 53 rows to see where the number of missing values goes to zero
x_asc_df[:53]

#Nice. Some realy important variables such as the distance to a metro station
#have a small number of missing values. We can drop the missing rows with no remorse after a quick check.

#Columns with missing values
df.columns[df.isnull().any()].tolist()


nuli = x_asc_df[0:51].transpose()
list_of_nulls = list(nuli.columns.values)
list_of_nulls

df_null = pd.DataFrame(df[list_of_nulls])
price = pd.DataFrame(df['price_doc'])
price_null = pd.concat([price,df_null], axis =1)
price_null

#cols = price_null.columns.tolist()
#cols
#cols.insert(0, cols.pop(cols.index('price_doc')))
#price_null = price_null.loc[:, cols]

corrmat2 = price_null.corr()
corrmat2


f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat2,linewidths=.1)

