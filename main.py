# Imports
import pandas as pd
import numpy as np
import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn import metrics
from sklearn.impute import SimpleImputer
import warnings

# Silence future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Read in data
train = pd.read_csv('bookings_train.csv')
test = pd.read_csv('bookings_test_solutions.csv')

train['set'] = 'train'
test['set'] = 'test'

data = pd.concat([train, test])

# Feature Engineering
data.rename(columns={'arrival_date_day_of_month': 'day',
                     'arrival_date_year': 'year',
                     'arrival_date_month': 'month',
                     'arrival_date_week_number': 'weeknum'},
             inplace=True)

# Convert month to number
data['month'] = data['month'].apply(lambda x: datetime.strptime(x, "%B").month)

# Create date
data['date'] = pd.to_datetime(data[['year', 'month', 'day']],
                                      format="%Y%B%d")

# Extract day of week
data['weekday'] = data['date'].dt.dayofweek

# Binary: Customer got reserved room
data['got_reserved_room'] = np.multiply((data['reserved_room_type'] == data['assigned_room_type']), 1)

# Total visitors
data['total_visitors'] = data['adults'] + data['children'] + data['babies']

# Convert types to category
for col in ['meal', 'country', 'market_segment', 'reserved_room_type', 'assigned_room_type',
            'deposit_type', 'customer_type', 'year']:
    data[col] = data[col].astype('category')

# Drop unnecessary columns
data.drop(columns=['date', 'babies', 'day', 'days_in_waiting_list'], inplace=True)

# Create train/test split from data files
train = data.loc[data['set'] == 'train']
test = data.loc[data['set'] == 'test']

y_train = train['is_cancelled']
X_train = train.drop('is_cancelled', axis=1)
y_test = test['is_cancelled']
X_test = test.drop('is_cancelled', axis=1)

# Drop the set column
X_train.drop(columns=['set'], inplace=True)
X_test.drop(columns=['set'], inplace=True)

# Remove seemingly wrong observations from train set
X_train = X_train.loc[~((X_train['stays_in_weekend_nights'] == 0) & (X_train['stays_in_week_nights'] == 0))]
X_train = X_train.loc[X_train['adults'] != 0]
X_train = X_train.loc[X_train['adr'] >= 0]

# Define Variable types
cat_feats = list(X_train.columns[np.where(X_train.dtypes == 'category')])
cat_feats.remove('country')
num_feats = list(np.setdiff1d(X_train.columns, (cat_feats + ['month', 'weekday', 'weeknum', 'country'])))

# Cyclic transformation for time variables
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

# Create one-hot encoder
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
country_transformer = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=200)

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy='most_frequent')), ("scaler", StandardScaler())]
)

preproccessor = ColumnTransformer(
    transformers=[
        ("numerical", numeric_transformer, num_feats),
        ("categorical", categorical_transformer, cat_feats),
        ("country", country_transformer, ['country']),
        ("month_sin", sin_transformer(12), ["month"]),
        ("month_cos", cos_transformer(12), ["month"]),
        ("weekday_sin", sin_transformer(7), ["weekday"]),
        ("weekday_cos", cos_transformer(7), ["weekday"]),
        ("weeknum_sin", sin_transformer(53), ["weeknum"]),
        ("weeknum_cos", cos_transformer(53), ["weeknum"]),
    ],
)

# Fit the preprocessor to the training data
X_train_prepped = preproccessor.fit_transform(X_train)
X_test_prepped = preproccessor.transform(X_test)
