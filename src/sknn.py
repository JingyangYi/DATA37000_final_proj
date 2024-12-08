import os
import math
import numpy as np
import pandas as pd
# pip install rfit
import rfit
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
# from sklearn.tree import DecisionTreeClassifier
#
import warnings

warnings.filterwarnings("ignore")
# %%
train = pd.read_csv("../data/HousePricesAdv/train.csv", header=0)
test = pd.read_csv("../data/HousePricesAdv/test.csv", header=0)
# %%
ordinal_mappings = {
    'ExterQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'ExterCond': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'BsmtQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': 0},
    'BsmtCond': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': 0},
    'HeatingQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'KitchenQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'FireplaceQu': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': 0},
    'GarageQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': 0},
    'GarageCond': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0, 'NA': 0},
    'PoolQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0},
    'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
}


def modify_data(data, ordinal_mappings):
    for feature, mapping in ordinal_mappings.items():
        if feature in data.columns:
            data[feature] = data[feature].map(mapping)
    categorical_columns = data.select_dtypes(include=['object']).columns
    data_dropped = data.drop(columns=categorical_columns)

    # Handling NaN values
    # For numerical columns, replace NaN with the median value
    numerical_columns = data_dropped.select_dtypes(include=['float64', 'int64']).columns
    data_dropped[numerical_columns] = data_dropped[numerical_columns].fillna(
        data_dropped[numerical_columns].median()
    )

    # For ordinal features that were mapped, replace NaN with 0 (unknown or missing)
    for feature in ordinal_mappings.keys():
        if feature in data_dropped.columns:
            data_dropped[feature] = data_dropped[feature].fillna(0)

    # Transform year variables into age-related features
    if 'YearBuilt' in data_dropped.columns:
        data_dropped['AgeBuilt'] = data_dropped['YrSold'] - data_dropped['YearBuilt']
    if 'YearRemodAdd' in data_dropped.columns:
        data_dropped['AgeRemodeled'] = data_dropped['YrSold'] - data_dropped['YearRemodAdd']

    # Drop the original year variables
    data_dropped = data_dropped.drop(columns=['YearBuilt', 'YearRemodAdd', 'YrSold'])

    return data_dropped


train = modify_data(train, ordinal_mappings)
test = modify_data(test, ordinal_mappings)


# %%
class sknn:
    '''
    Scaling k-NN model
    v3 - Enhanced with saving intermediate results and tracking best scaling factors.
    '''

    def __init__(self,
                 data_x,
                 data_y,
                 resFilePfx='results',
                 classifier=True,
                 k=7,
                 kmax=33,
                 zscale=True,
                 caleExpos_init=(),
                 scales_init=(),
                 ttsplit=0.5,
                 max_iter=100,
                 seed=1,
                 scoredigits=6,
                 learning_rate_init=0.1,
                 atol=1e-8):

        self.__classifierTF = classifier
        self.k = k
        self.__kmax = kmax
        self.max_iter = max_iter
        self.__seed = seed
        self.__scoredigits = scoredigits
        self.__learning_rate_init = abs(learning_rate_init)
        self.learning_rate = abs(learning_rate_init)
        self.__atol = atol

        self.data_x = data_x
        self.data_xz = data_x
        self.zscaleTF = zscale
        if self.zscaleTF:
            self.zXform()

        self.data_y = data_y
        self.__ttsplit = ttsplit if (ttsplit >= 0 and ttsplit <= 1) else 0.5
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.__xdim = 0
        self.traintestsplit()

        self.__vector0 = np.zeros(self.__xdim)

        self.__scaleExpos = []
        self.__scaleFactors = None
        self.__setExpos2Scales([])

        if self.__classifierTF:
            self.__knnmodels = [np.nan, np.nan] + [
                KNeighborsClassifier(n_neighbors=i, weights='uniform').fit(self.X_train, self.y_train)
                for i in range(2, self.__kmax + 1)
            ]
        else:
            self.__knnmodels = [np.nan, np.nan] + [
                KNeighborsRegressor(n_neighbors=i, weights='uniform').fit(self.X_train, self.y_train)
                for i in range(2, self.__kmax + 1)
            ]
        self.benchmarkScores = [np.nan, np.nan] + [
            round(x.score(self.X_test, self.y_test), self.__scoredigits) for x in self.__knnmodels[2:]
        ]
        print(f'Basic k-NN scores for different k-values: {repr(self.benchmarkScores)}')

        # History list to store intermediate results
        # Each entry: (iteration, scaleExpos, scaleFactors, gradient, train_score, test_score)
        self.history = []

        # Track best scaling factors
        self.best_score = -np.inf
        self.best_scaleFactors = None
        self.best_scaleExpos = None

    def zXform(self):
        scaler = StandardScaler()
        self.data_xz = scaler.fit_transform(self.data_x)
        return

    def traintestsplit(self):
        dy = self.data_y.values if (
                    isinstance(self.data_y, pd.Series) or isinstance(self.data_y, pd.DataFrame)) else self.data_y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_xz, dy,
                                                                                test_size=self.__ttsplit,
                                                                                random_state=self.__seed)
        _, self.__xdim = self.X_test.shape
        return

    def __setExpos2Scales(self, expos=[]):
        if (len(expos) != self.__xdim):
            self.__scaleExpos = np.zeros(self.__xdim)
            if self.__xdim > 1:
                self.__scaleExpos[0] = 1
                self.__scaleExpos[1] = -1
        else:
            self.__scaleExpos = expos
        self.__scaleFactors = np.array([math.exp(i) for i in self.__scaleExpos])
        return

    def __shiftCenter(self, expos=[]):
        return expos.copy() - expos.sum() / len(expos) if len(expos) > 1 else expos.copy()

    def __evalGradients(self, learning_rate=0, use='test'):
        grad = np.array([self.__eval1Gradient(i, learning_rate, use=use) for i in range(self.__xdim)])
        return grad

    def __eval1Gradient(self, i, learning_rate=0, use='test'):
        thescale = self.__scaleExpos[i]
        thestep = max(learning_rate, self.learning_rate, abs(thescale) * self.learning_rate)
        maxexpos = self.__scaleExpos.copy()
        maxexpos[i] += thestep / 2
        minexpos = self.__scaleExpos.copy()
        minexpos[i] -= thestep / 2
        slope = (self.scorethis(scaleExpos=maxexpos, use=use) - self.scorethis(scaleExpos=minexpos, use=use)) / thestep
        return slope

    def __setNewExposFromGrad(self, grad=()):
        grad = self.__shiftCenter(grad)
        if np.allclose(grad, self.__vector0, atol=self.__atol):
            print(
                f"Gradient is near zero, stopping optimization.\n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \ntrain score={self.scorethis(use='train')}, test score={self.scorethis(use='test')}\n")
            return False
        norm = np.sqrt(np.dot(grad, grad))
        deltaexpos = grad / norm * self.learning_rate
        self.__scaleExpos += deltaexpos
        self.__setExpos2Scales(self.__scaleExpos)
        return True

    def optimize(self, scaleExpos_init=(), maxiter=0, learning_rate=0, save_interval=100):
        maxi = max(self.max_iter, maxiter, 1000)
        if (len(scaleExpos_init) == self.__xdim):
            self.__scaleExpos = scaleExpos_init
            self.__setExpos2Scales(self.__scaleExpos)

        print(
            f"Begin optimization: \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \ntrain score={self.scorethis(use='train')}, test score={self.scorethis(use='test')}, \nmaxi={maxi}, k={self.k}, learning_rate={self.learning_rate}\n")

        for i in range(maxi):
            grad = self.__evalGradients(learning_rate, use='train')
            result = self.__setNewExposFromGrad(grad)

            # Check performance on train/test
            train_score = self.scorethis(use='train')
            test_score = self.scorethis(use='test')

            # Save intermediate results every save_interval steps or on first iteration
            if (i % save_interval == 0) or (i == maxi - 1) or (i < 10):
                self.history.append({
                    'iteration': i,
                    'scaleExpos': self.__scaleExpos.copy(),
                    'scaleFactors': self.__scaleFactors.copy(),
                    'gradient': grad.copy(),
                    'train_score': train_score,
                    'test_score': test_score
                })
                print(
                    f"i={i}, |grad|^2={np.dot(grad, grad):.6f}, train_score={train_score:.6f}, test_score={test_score:.6f}")

            # Track best model
            if test_score > self.best_score:
                self.best_score = test_score
                self.best_scaleFactors = self.__scaleFactors.copy()
                self.best_scaleExpos = self.__scaleExpos.copy()

            if not result:
                print("Stopping early due to negligible gradient.")
                break

        print(
            f"Optimization finished.\nBest test score={self.best_score}, \nbest_scaleFactors={self.best_scaleFactors}, \nbest_scaleExpos={self.best_scaleExpos}\n")

    def scorethis(self, scaleExpos=[], scaleFactors=[], use='test'):
        if len(scaleExpos) == self.__xdim:
            self.__setExpos2Scales(self.__shiftCenter(scaleExpos))
        sfactors = self.__scaleFactors.copy()
        self.__knnmodels[self.k].fit(sfactors * self.X_train, self.y_train)
        newscore = self.__knnmodels[self.k].score(sfactors * self.X_train, self.y_train) if use == 'train' else \
        self.__knnmodels[self.k].score(sfactors * self.X_test, self.y_test)
        return newscore


model = sknn(train.drop(columns=['SalePrice']), train['SalePrice'], classifier=False, k=5, max_iter=500)
model.optimize(save_interval=100)

# Convert model.history (list of dicts) to a DataFrame
history_df = pd.DataFrame(model.history)

history_df['scaleExpos'] = history_df['scaleExpos'].apply(lambda x: ','.join(map(str, x)))
history_df['scaleFactors'] = history_df['scaleFactors'].apply(lambda x: ','.join(map(str, x)))
history_df['gradient'] = history_df['gradient'].apply(lambda x: ','.join(map(str, x)))

# %%
history_df.to_csv('../model_history.csv', index=False)
print("History saved to ../model_history.csv")
# %%
best_scale_factors = model.best_scaleFactors

# Train a Random Forest regressor on the same training data
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(model.X_train, model.y_train)

# Get feature importances from the random forest
rf_importances = rf.feature_importances_

if isinstance(model.data_x, pd.DataFrame):
    feature_names = model.data_x.columns
else:
    feature_names = [f"Feature_{i}" for i in range(model.__xdim)]

# Combine the results into a DataFrame for easier comparison
comparison_df = pd.DataFrame({
    'Feature': feature_names,
    'Best_Scale_Factor': best_scale_factors,
    'RF_Feature_Importance': rf_importances
})

comparison_df.sort_values('RF_Feature_Importance', ascending=False, inplace=True)
print("-----" * 10)
print("-----" * 10)
print("Comparison of sknn-derived scale factors and RF feature importances:")
print(comparison_df)