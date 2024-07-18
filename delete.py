# %%
import pandas as pd
# %%
PATH = 'outputs/scores/ml_emails.csv_test_0.8_train_seed_333.csv'
# %%
df = pd.read_csv(PATH)
# %%
df.head()
# %%
print(df)
# %%
df.f1.idxmax()
# %%
df.iloc[df.f1.idxmax()][0]
# %%
