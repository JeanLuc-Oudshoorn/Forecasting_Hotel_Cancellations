# Imports
import pandas as pd
import numpy as np
import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
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

# Remove seemingly wrong observations from train set
train = train.loc[~((train['stays_in_weekend_nights'] == 0) & (train['stays_in_week_nights'] == 0))]
train = train.loc[train['adults'] != 0]
train = train.loc[train['adr'] >= 0]

# Split predictors and outcome
y_train = train['is_cancelled']
X_train = train.drop('is_cancelled', axis=1)
y_test = test['is_cancelled']
X_test = test.drop('is_cancelled', axis=1)

# Drop the set column
X_train.drop(columns=['set'], inplace=True)
X_test.drop(columns=['set'], inplace=True)

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

# Instantiate a LightGBM Classifier with pre-known hyperparameters
lgbm_estimator = lgb.LGBMClassifier(random_state=7,
                                    feature_fraction=22/len(X_train.columns),
                                    n_estimators=3600,
                                    min_child_samples=15,
                                    max_depth=11,
                                    learning_rate=0.019,
                                    n_jobs=-1,
                                    verbosity=1)

# Fit LightGBM
print("Fit LightGBM\n")
lgbm_estimator.fit(X_train_prepped, y_train)

# Instantiate Random Forest Classifier with pre-known hyperparameters
rf_estimator = RandomForestClassifier(random_state=7,
                                      n_estimators=500,
                                      max_features=10)

# Fit Random Forest
print("Fit Random Forest\n")
rf_estimator.fit(X_train_prepped, y_train)

# Define hyperparameter grid
cb_param_dist = {
    'depth': [9, 10, 11],
    'learning_rate': [0.02, 0.03, 0.1],
    'l2_leaf_reg': [0, 1],
    'iterations': [300, 600]
}

# Instantiate CatBoost estimator
catboost_estimator = cb.CatBoostClassifier(random_seed=7)

# Create and fit a GridSearch object
print("Start Gridsearch CatBoost\n")
cb_model = GridSearchCV(catboost_estimator, param_grid=cb_param_dist, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)
cb_model.fit(X_train_prepped, y_train, verbose=False)

# Instantiate Voting Classifier
vc_model = VotingClassifier(estimators=[('cb', cb.CatBoostClassifier(**cb_model.best_params_, random_seed=7)),
                                        ('lgbm', lgbm_estimator),
                                        ('rf', rf_estimator)],
                            voting='soft')

# Fit voting classifier to the data
print("Fitting Voting Classifier\n")
vc_model.fit(X_train_prepped, y_train)

# Calculate test AUC and Accuracy
model = [cb_model, lgbm_estimator, rf_estimator, vc_model]

for m in model:
    auc_score = metrics.roc_auc_score(y_test, m.predict_proba(X_test_prepped)[::, 1])
    print("{model} AUC scores is: {auc}\n".format(model=m, auc=auc_score))

for m in model:
    accuracy = metrics.accuracy_score(y_test, m.predict(X_test_prepped))
    print("{model} Accuracy is: {acc}\n".format(model=m, acc=accuracy))

# Plot ROC-curve
fig, ax = plt.subplots()
plt.grid(visible=True)

for m in model:
    y_pred = m.predict_proba(X_test_prepped)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred, pos_label='yes')
    plt.plot(fpr, tpr, alpha=0.5)

plt.legend(['CatBoost', 'LightGBM', 'Random Forest', 'Voting Clf.'])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='grey')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ROC curve")
plt.savefig('hotel_roc_curve.png')
plt.show()

# Calculate F1-scores
for m in model:
    f1 = metrics.f1_score(y_test, m.predict(X_test_prepped), pos_label='yes')
    print("{model} F1 score is: {acc}\n".format(model=m, acc=f1))

# Plot PR-curve
plt.grid(visible=True)

for m in model:
    y_pred = m.predict_proba(X_test_prepped)[::, 1]
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred, pos_label='yes')
    plt.plot(precision, recall, alpha=0.5)

plt.legend(['CatBoost', 'LightGBM', 'Random Forest', 'Voting Clf.'])
plt.ylabel('Recall')
plt.xlabel('Precision')
plt.title("PR Curve")
plt.savefig('hotel_pr_curve.png')
plt.show()
