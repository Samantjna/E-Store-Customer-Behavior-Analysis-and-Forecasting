import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 20)

# data = pd.read_excel('Adidas US Sales Datasets.xlsx')
# data.to_csv('Adidas US Sales Datasets.csv', index=False)

data = pd.read_csv('Adidas US Sales Datasets.csv')


#isfiltruoojam pagal lyti
data['Gender'] = data['Product'].apply(lambda x: 'Men' if "Men's" in x else ('Women' if "Women's" in x else 'Other'))

# data.to_csv('Adidas US Sales Datasetss_.csv')
# print(data.head())

df = pd.read_csv('Adidas US Sales Datasetss_.csv')

# df['Rating'] = np.random.randint(1, 11, size=len(df))
# df.to_csv('Adidas US Sales Datasets_.csv')
# print(df.head())


df['Rating'] = np.random.randint(1, 7, size=len(df))

# 60 proc tikimybe, generuojam nuo 7-10
df.loc[np.random.choice(df.index, size=int(len(df) * 0.6), replace=False), 'Rating'] = np.random.randint(7, 11)

# print(df.head())
# df.to_csv('Adidas US Sales Datasets_final.csv', index=False)

df = pd.read_csv('Adidas US Sales Datasets_final.csv')
print(df.head())
