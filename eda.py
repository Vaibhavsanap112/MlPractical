#assignment No:01

import pandas as pd
import numpy as np
df = pd.read_csv("1.csv")
df.info()
df.head()
df.tail()
df.dtypes
drop_colunm = ['Edition Statement','Corporate Author','Corporate Contributors','Former owner','Engraver','Shelfmarks']

df.drop(drop_colunm, inplace=True, axis=1)
df.head()
df.describe()
df.info()
df['Identifier'].is_unique
df = df.set_index('Identifier')
df.head()
df.loc[472]
extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
extr.head()
df.dtypes
df.loc[1905:, 'Date of Publication'].head(10)

df['Date of Publication'] = pd.to_numeric(extr)
df['Date of Publication'].dtype
df['Date of Publication'].isnull().sum() / len(df)
new_column_names = {
    "Place of Publication": "Publication Place",
    "Flickr URl": "URL"
}

data = df.rename(columns=new_column_names)
data.columns
data.info()
data.head()
import matplotlib.pyplot as plt
df.plot(kind='hist')
df['Publisher'].value_counts().plot(kind='bar' )
df.plot(kind='bar')