import pandas as pd
import numpy as np

df = pd.read_csv('retail_sales_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Basic stats
print("ACTUAL VALUES FOR REPORT VERIFICATION")
print("="*50)
print(f"Total Transaksi: {len(df)}")
print(f"Total Revenue: ${df['Total Amount'].sum():,.0f}")
print(f"Avg Transaction: ${df['Total Amount'].mean():.2f}")
print(f"Unique Customers: {df['Customer ID'].nunique()}")

# Category stats
print("\nCATEGORY REVENUE:")
cat_rev = df.groupby('Product Category')['Total Amount'].agg(['sum','mean','count'])
for idx, row in cat_rev.iterrows():
    pct = row['sum']/df['Total Amount'].sum()*100
    print(f"  {idx}: ${row['sum']:,.0f} ({pct:.1f}%), Avg ${row['mean']:.2f}, Count {row['count']}")

# Gender stats  
print("\nGENDER REVENUE:")
gen_rev = df.groupby('Gender')['Total Amount'].agg(['sum','mean','count'])
for idx, row in gen_rev.iterrows():
    print(f"  {idx}: ${row['sum']:,.0f}, Avg ${row['mean']:.2f}, Count {row['count']}")

# Monthly
print("\nMONTHLY REVENUE:")
monthly = df.groupby(df['Date'].dt.month)['Total Amount'].sum()
print(f"  Best Month: {monthly.idxmax()} (${monthly.max():,.0f})")
print(f"  Lowest Month: {monthly.idxmin()} (${monthly.min():,.0f})")

print("\n" + "="*50)
