# Imports
import pandas as pd
import numpy as np
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn import metrics
from sklearn.impute import SimpleImputer
from mlxtend.classifier import StackingCVClassifier
import warnings
import csv

# Silence future warnings
warnings.simplefilter(action='ignore')

# Read in data
train = pd.read_csv('bookings_train.csv')
test = pd.read_csv('bookings_test_solutions.csv')

train['set'] = 'train'
test['set'] = 'test'

data = pd.concat([train, test])

###############################################################################
#                  1. Feature Engineering & Preprocessing                     #
###############################################################################


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
y_train = np.multiply(train['is_cancelled'] == 'yes', 1)
X_train = train.drop('is_cancelled', axis=1)
y_test = np.multiply(test['is_cancelled'] == 'yes', 1)
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

print("Preprocessing Finished\n")

###############################################################################
#                         2. Instantiate Classifiers                          #
###############################################################################


# Instantiate a LightGBM Classifier with pre-known hyperparameters
lgbm_estimator = lgb.LGBMClassifier(random_state=7,
                                    colsample_bytree=0.88,
                                    n_estimators=3600,
                                    min_child_samples=15,
                                    max_depth=11,
                                    learning_rate=0.019,
                                    n_jobs=-1,
                                    verbosity=-10)


# Instantiate Random Forest Classifier with pre-known hyperparameters
rf_estimator = RandomForestClassifier(random_state=7,
                                      n_estimators=500,
                                      max_features=10,
                                      verbose=False,
                                      n_jobs=-1)


# Instantiate CatBoost estimator
catboost_estimator = cb.CatBoostClassifier(random_seed=7,
                                           depth=11,
                                           learning_rate=0.03,
                                           l2_leaf_reg=1,
                                           iterations=600,
                                           thread_count=-1,
                                           verbose=False)


# Instantiate XGBoost estimator
xgb_estimator = xgb.XGBClassifier(verbosity=0,
                                  seed=7,
                                  nthread=-1,
                                  learning_rate=0.02,
                                  max_depth=11,
                                  n_estimators=700,
                                  min_child_weight=10,
                                  colsample_bytree=0.88)

# Instantiate Voting Classifier
vc_estimator = VotingClassifier(estimators=[('lgbm', lgbm_estimator),
                                            ('rf', rf_estimator),
                                            ('cb', catboost_estimator),
                                            ('xgb', xgb_estimator)],
                                voting='soft',
                                n_jobs=-1,
                                verbose=0)


# Instantiate the StackingCV classifier
meta_classifier = RandomForestClassifier(random_state=7)

sclf_base = StackingCVClassifier(classifiers=[lgbm_estimator, rf_estimator, catboost_estimator, xgb_estimator],
                                 cv=4,
                                 shuffle=False,
                                 use_probas=True,
                                 n_jobs=-1,
                                 random_state=7,
                                 meta_classifier=meta_classifier)

# Define parameter grid
params = {"meta_classifier__max_depth": [4, 5],
          "meta_classifier__min_samples_leaf": [30, 40]}


# Initialize GridSearchCV
sclf_estimator = GridSearchCV(estimator=sclf_base,
                              param_grid=params,
                              cv=4,
                              scoring="accuracy",
                              verbose=10,
                              n_jobs=-1)


# Create list to store classifiers
estimators = [lgbm_estimator, catboost_estimator, rf_estimator, xgb_estimator, vc_estimator, sclf_estimator]
names = ['LGBM', 'CB', 'RF', 'XGB', 'Voting', 'Stack']
classifiers = dict(zip(names, estimators))

print("All Estimators Instantiated\n")

###############################################################################
#                           3. Training Classifiers                           #
###############################################################################

# Train classifiers
for key in classifiers:

    # Get classifier
    start = datetime.now()
    classifier = classifiers[key]

    # Fit classifier
    print("Now fitting: {}\n".format(key))
    classifier.fit(X_train_prepped, y_train)

    # Save fitted classifier
    end = datetime.now()
    elapsed_time = end - start

    print("Now Saving: {model}\n Elapsed Time: {time}\n".format(model=key, time=elapsed_time))
    classifiers[key] = classifier

print("Best Hyperparameters for Stacking:\n", sclf_estimator.best_params_)

print("All Estimators Fitted\n")

###############################################################################
#                           4. Making predictions                             #
###############################################################################

# Get results
results = pd.DataFrame()

for key in classifiers:
    # Make prediction on test set
    y_pred = classifiers[key].predict_proba(X_test_prepped)[:, 1]

    # Save results in pandas dataframe object
    print("Now saving predictions: {}\n".format(key))
    results[f"{key}"] = y_pred

# Add the test set to the results object
results["Target"] = y_test

###############################################################################
#                           5. Visualizing results                            #
###############################################################################

# Probability Distributions Figure
# Set graph style
sns.set(font_scale=1)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})

# Plot probability distributions
f, ax = plt.subplots(figsize=(13, 4), nrows=1, ncols=5)

for key, counter in zip(classifiers, range(5)):
    # Get predictions
    y_pred = results[key]

    # Get AUC
    auc = metrics.roc_auc_score(y_test, y_pred)
    textstr = f"AUC: {auc:.3f}"

    print("Now plotting: {}\n".format(key))

    # Plot false distribution
    false_pred = results[results["Target"] == 0]
    sns.distplot(false_pred[key], hist=True, kde=False,
                 bins=int(25), color='red',
                 hist_kws={'edgecolor': 'black'}, ax=ax[counter])

    # Plot true distribution
    true_pred = results[results["Target"] == 1]
    sns.distplot(results[key], hist=True, kde=False,
                 bins=int(25), color='green',
                 hist_kws={'edgecolor': 'black'}, ax=ax[counter])

    # These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # Place a text box in upper left in axes coords
    ax[counter].text(0.05, 0.95, textstr, transform=ax[counter].transAxes, fontsize=14,
                     verticalalignment="top", bbox=props)

    # Set axis limits and labels
    ax[counter].set_title(f"{key} Distribution")
    ax[counter].set_xlim(0, 1)
    ax[counter].set_xlabel("Probability")

# Tight layout
plt.tight_layout()

# Save Figure
plt.savefig("rf_meta_prob_distr.png", dpi=600)

###############################################################################
#                         6. Evaluate Classifiers                             #
###############################################################################

for key in classifiers:
    auc_score = metrics.roc_auc_score(y_test, classifiers[key].predict_proba(X_test_prepped)[::, 1])
    print("{model} AUC score is: {auc}\n".format(model=key, auc=auc_score))

for key in classifiers:
    accuracy_score = metrics.accuracy_score(y_test, classifiers[key].predict(X_test_prepped))
    print("{model} Accuracy score is: {acc}\n".format(model=key, acc=accuracy_score))

###############################################################################
#                            7. Plot ROC-curve                                #
###############################################################################

# Plot ROC-curve
fig, ax = plt.subplots()
plt.grid(visible=True)

for key in classifiers:
    y_pred = classifiers[key].predict_proba(X_test_prepped)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred, pos_label=1)
    plt.plot(fpr, tpr, alpha=0.5)

plt.legend(['LightGBM', 'CatBoost', 'Random Forest', 'XGBoost', 'Voting Clf.', 'Stacking Clf.'])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='grey')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ROC curve")
plt.savefig('rf_meta_roc_curve.png', dpi=600)
plt.show()

###############################################################################
#                            8. Write Results                                 #
###############################################################################

header = ['Algorithm', 'ROC AUC Score', 'Accuracy', 'Hyperparameters']

# Open file in write mode
with open('results.csv', 'w') as f:

    # Create a writer
    writer = csv.writer(f)

    # Write header to file
    writer.writerow(header)

    for key in classifiers:

        estimator_string = re.search('(\w+)\s?', str(classifiers[key])).group(1)

        upd_string = estimator_string

        if key == 'Stack':
            estimator_string = re.search('estimator=(\w+)\s?', str(classifiers[key])).group(1)
            meta_string = re.search('meta_classifier=(\w+)\s?', str(classifiers[key])).group(1)
            upd_string = estimator_string + ' with ' + meta_string

        roc_auc_score = np.round(metrics.roc_auc_score(y_test, classifiers[key].predict_proba(X_test_prepped)[::, 1]),
                                 6)

        accuracy_score = np.round(metrics.accuracy_score(y_test, classifiers[key].predict(X_test_prepped)), 6)

        if key == 'Voting':
            params = "None"
        elif key == 'Stack':
            params = classifiers[key].best_params_
        else:
            params = classifiers[key].get_params()

        tmp_row = [upd_string, roc_auc_score, accuracy_score, params]

        writer.writerow(tmp_row)

print("Results Written\n")
