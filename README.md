# Favorita sales forecasting

This is a data science project based on the Kaggle competition ["Store Sales - Time Series Forecasting"](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/), which concerns the 'Favorita' grocery stores, an Ecuadorian chain. 

The dataset (found in the `input` folder) contains daily sales data from 2012 to 2017. The goal is to predict the following 15 days of sales in 54 stores, across 33 product families. Accuracy of predictions is scored using RMSLE.

## The data

The `input` folder contains the following datasets.

1. `train.csv` contains a time series of various features, with sales data. This is intended to be used as training data.
2. `test.csv` contains a time series of the same features as `train.csv` for the next 15 days, but without sales data. The goal is to predict the sales data.
3. `stores.csv` provides information about the location and type of each store, which could allow us to identify trends based on geographic location.
4. `oil.csv` contains the daily oil price, which is known to have a big influence on Ecuador's economy.
5. `holidays_events.csv` contains information on important holidays and events in Ecuador, which could help explain anomalous days on which sales spike or drop.
6. `transactions.csv` contains the number of transactions that each store makes on a given day, which for example could be used in tandem with the sales figures to determine average spend per sale.

In addition, we have the following knowledge.

- Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
- A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.

## Methodology

Below is a brief summary of the methodology. A more detailed methodology, complete with code snippets and plots, can be found in the notebook `favorita-sales-forecasting.ipynb`.

The initial goal is to provide predictions using only the `train.csv` and `test.csv` datasets. We apply the following techniques to achieve this.

1. **Data cleaning.** This includes converting dates from strings to `datetime64` data types and dropping the `id` column, which is just a copy of the index.
2. **Anomaly detection.** We visualise the data with a scatter plot and remove rows with anomalous data. We focus particularly on the weeks following the earthquake, which we know caused abnormal sales. We do not expect an earthquake in the next 15 days, so this should not factor into our predictions.
3. **Target encoding.** We use an m-estimate encoding on the `family` feature, turning product families into weighted means of the sales figures within the family. We also encode dates using a simple integer encoding.
4. **Autoregression.** Since we are working with a time series, we expect some degree of autoregression. We test for autocorrelation and select the appropriate amount of lag based on our finding. Since this turns out to be 7 days, we add a `sales_last_week` feature.
5. **Train-test splitting.** We split the data in `train.csv` into a training set and a testing set. We do not split randomly, because we are working with a time series and therefore test data needs to come after training data, chronologically.
6. **Walk-forward validation.** When testing, we need to be careful to remove data in the `sales_last_week` column except for the first week of test data. Otherwise, our test data will contain information about test sales, which will make our predictions misleadingly optimistic. Instead, we predict week-by-week, filling in the `sales_last_week` column as we go, based on the predicted sales for the previous week.
7. **Model selection.** We use linear regression as our baseline model and test this against a random forest model, which performs well. 
8. **Hyperparameter tuning.** We tune the hyperparameters of the random forest model, choosing the ones that result in an optimal RMSLE.

## Results

After running the script `favorita_forecast.py`, the prediction data will be outputted to the file `predictions.csv`.

Our first pass gave an RMSLE of 0.50829 in testing, while our predictions resulted in an RMSLE of 0.52590 when submitted to the Kaggle competition. 
