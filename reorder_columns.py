import pandas as pd

# Load the data
df = pd.read_csv('./dataset/BTCUSD/BTCUSD_1m.csv')

# Current columns
print("Current columns:", list(df.columns))

# Reorder columns so Close is last
cols = list(df.columns)
cols.remove('Close')
cols.remove('date')
df = df[['date'] + cols + ['Close']]

print("New columns:", list(df.columns))

# Save back
df.to_csv('./dataset/BTCUSD/BTCUSD_1m.csv', index=False)

print("Columns reordered successfully!")