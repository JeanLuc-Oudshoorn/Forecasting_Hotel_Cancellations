# Imports
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from mlxtend.classifier import StackingCVClassifier
import warnings

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

print("Preprocessing Finished\n")

###############################################################################
#                         2. Instantiate Classifiers                          #
###############################################################################

# Instantiate a LightGBM Classifier with pre-known hyperparameters
lgbm_estimator = lgb.LGBMClassifier(random_state=7,
                                    n_estimators=3600,
                                    min_child_samples=15,
                                    max_depth=11,
                                    learning_rate=0.019,
                                    n_jobs=-1,
                                    verbosity=-100)

# Instantiate a Catboost Classifier with pre-known hyperparameters
cb_estimator = cb.CatBoostClassifier(random_seed=7,
                                     thread_count=-1,
                                     learning_rate=0.03,
                                     depth=11,
                                     l2_leaf_reg=1,
                                     iterations=600,
                                     verbose=False)

# Instantiate Random Forest Classifier with pre-known hyperparameters
rf_estimator = RandomForestClassifier(random_state=7,
                                      n_estimators=500,
                                      max_features=10)

# Instantiate Multilayer Perceptron Classifier with pre-known hyperparameters
mlp_estimator = MLPClassifier(random_state=7,
                              max_iter=400,
                              hidden_layer_sizes=(8, 8, 8),
                              learning_rate_init=0.0007,
                              alpha=0.00005)

# Instantiate Voting Classifier
vclf_estimator = VotingClassifier(estimators=[('lgbm', lgbm_estimator),
                                              ('cb', cb_estimator),
                                              ('rf', rf_estimator),
                                              ('mlp', mlp_estimator)],
                                  voting='soft',
                                  verbose=False)

# Initializing the StackingCV classifier
sclf_estimator = StackingCVClassifier(classifiers=[lgbm_estimator, cb_estimator, rf_estimator, mlp_estimator],
                                      shuffle=False,
                                      use_probas=True,
                                      cv=5,
                                      meta_classifier=LogisticRegression(random_state=7),
                                      verbose=0,
                                      n_jobs=-1)


# Create list to store classifiers
classifiers = {"LGBM": lgbm_estimator,
               "CB": cb_estimator,
               "RF": rf_estimator,
               "MLP": mlp_estimator,
               "Voting": vclf_estimator,
               "Stack": sclf_estimator}

print("All Estimators Instantiated\n")

###############################################################################
#                            3. Training Classifier                           #
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
    false_pred = results[results["Target"] == 'no']
    sns.distplot(false_pred[key], hist=True, kde=False,
                 bins=int(25), color='red',
                 hist_kws={'edgecolor': 'black'}, ax=ax[counter])

    # Plot true distribution
    true_pred = results[results["Target"] == 'yes']
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
plt.savefig("comparison_prob_distr.png", dpi=1080)

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

plt.legend(['LightGBM', 'CatBoost', 'Random Forest', 'MLP', 'Voting Clf.', 'Stacking Clf.'])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='grey')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ROC curve")
plt.savefig('comparison_roc_curve.png')
plt.show()
