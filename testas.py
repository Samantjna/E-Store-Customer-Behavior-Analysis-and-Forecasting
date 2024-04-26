import pandas as pd
pd.set_option('display.max_columns', 20)

# data = pd.read_excel('Adidas US Sales Datasets.xlsx')
# data.to_csv('Adidas US Sales Datasets.csv', index=False)

df = pd.read_csv('Adidas US Sales Datasets.csv')


#isfiltruoojam pagal lyti
df['Gender'] = df['Product'].apply(lambda x: 'Men' if "Men's" in x else ('Women' if "Women's" in x else 'Other'))

print(df)