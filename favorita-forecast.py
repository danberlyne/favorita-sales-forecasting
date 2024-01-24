import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from category_encoders import MEstimateEncoder

# Load data
df_train = pd.read_csv("../input/favorita-store-sales/train.csv")
df_test = pd.read_csv("../input/favorita-store-sales/test.csv")
df_stores = pd.read_csv("../input/favorita-store-sales/stores.csv")
df_oil = pd.read_csv("../input/favorita-store-sales/oil.csv")
df_holidays = pd.read_csv("../input/favorita-store-sales/holidays_events.csv")
df_transactions = pd.read_csv("../input/favorita-store-sales/transactions.csv")

# Clean data
df_train['date'] = pd.to_datetime(df_train['date'], format='%Y-%m-%d')
df_train = df_train.loc[:,'date':'onpromotion']
df_test['date'] = pd.to_datetime(df_test['date'], format='%Y-%m-%d')
X_test = df_test.loc[:,'date':'onpromotion']

# Drop anomalies
X = df_train.copy()
X = X.drop(X.loc[X.date.between('2016-04-16', '2016-05-13')].index)
X = X.drop(X.loc[X.date.between('2016-10-03', '2016-10-09')].index)
y = X.pop('sales')

# Target encoding
date_enc = LabelEncoder()
X.date = date_enc.fit_transform(X.date)
X_test.date = date_enc.fit_transform(X_test.date)
X_test.date += 1649 # Align dates with `X_train`
X_encode = X.sample(frac=0.03, random_state=23)
y_encode = y[X_encode.index]
X_train = X.drop(X_encode.index)
y_train = y[X_train.index]
encoder = MEstimateEncoder(cols=['family'], m=10.0)
encoder.fit(X_encode, y_encode)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

# Autoregression feature
X_train['sales_last_week'] = y_train.shift(33*54*7)
X_train.dropna(inplace=True)
y_train = y_train[X_train.index]
X_test['sales_last_week'] = pd.Series([0 for i in X_test.index], index=X_test.index, name='sales_last_week')
X_test.iloc[:33*54*7, -1] = y_train.iloc[-33*54*7:]

# Set up model
reg_model = RandomForestRegressor(min_samples_leaf=10, max_depth=25, max_leaf_nodes=70000, max_features=0.9, n_jobs=8, random_state=23)
reg_model.fit(X_train, y_train)

# Make predictions
predictions = reg_model.predict(X_test.iloc[:33*54*7])
X_test.iloc[33*54*7 : 33*54*7*2, -1] = pd.Series(predictions[:33*54*7], index=X_test.iloc[33*54*7 : 33*54*7*2, -1].index)
predictions = np.concatenate((predictions, reg_model.predict(X_test.iloc[33*54*7 : 33*54*7*2])))
X_test.iloc[33*54*7*2 : 33*54*16, -1] = pd.Series(predictions[33*54*7:33*54*9], index=X_test.iloc[33*54*7*2 : 33*54*16, -1].index)
predictions = np.concatenate((predictions, reg_model.predict(X_test.iloc[33*54*7*2 : 33*54*16])))

# Reformat predictions and save data
df_pred = pd.DataFrame({'id': df_test.id, 'sales': predictions})
df_pred.to_csv("predictions.csv", index=False)