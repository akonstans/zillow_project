import wrangle as w
import new_lib as nl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from math import sqrt
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import warnings
warnings.filterwarnings('ignore')

def select_kbest(X_train, y_train, stat_test, k_value):
    f_selector = SelectKBest(stat_test, k = k_value)
    f_selector.fit(X_train, y_train)
    f_select_mask = f_selector.get_support()
    return X_train.iloc[:,f_select_mask].head()

def rfe_ranks(n, X_train, y_train):
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select = n)
    rfe.fit(X_train, y_train)
    ranks = rfe.ranking_
    columns = X_train.columns.tolist()
    feature_ranks = pd.DataFrame({'ranking': ranks, 
                              'feature': columns})
    return feature_ranks.sort_values('ranking').reset_index().drop(columns = 'index')

    
def corr_test(cont_var, train_scaled, train):
    for col in cont_var:
        corr, p = stats.pearsonr(train_scaled[col], train['tax_value'])
    print(f'The correlation between {col} and tax_value is: {round(corr *100, 2)}\n')
    # simple correlation test for all continuous variables

def bed_plot(df):
    sns.barplot(x='bedrooms', y='tax_value', data= df, palette = 'dark')
    plt.xlabel('Bedrooms')
    plt.ylabel('Home Value')
    plt.title('Home Value by Bedroom Number')
    
def bath_plot(df):
    sns.barplot(x='bathrooms', y='tax_value', data=df, palette = 'pastel')
    plt.xlabel('Bathrooms')
    plt.ylabel('Home Values')
    plt.title('Property Value by Number of Bathrooms')

def county_plot(df):
    sns.boxplot(x='fips', y='tax_value',data=df, palette = ['darkorange', 'steelblue', 'forestgreen'])
    plt.xlabel('FIPS Code')
    plt.ylabel('Home Price')
    plt.title('Home Price by Federal County')
    plt.show()
# Finding home price by county and comparing them

def hist_plot(df):
    la = df[df['fips']==6037]['tax_value']
    orange = df[df['fips']==6059]['tax_value']
    ventura = df[df['fips']==6111]['tax_value']
    plt.hist(x = la, color = 'gold', alpha = .4, edgecolor = 'black', label = 'Los Angeles')
    plt.hist(x = orange, color = 'darkorange', alpha = .5, edgecolor = 'black', label = 'Orange')
    plt.hist(x = ventura, color = 'mediumblue', alpha = .5, edgecolor = 'black', label = 'Ventura')
    plt.xlabel('Home Value')
    plt.ylabel('Properties')
    plt.title('Property Values By County')
    plt.legend()
    plt.show()
# Layering previous histograms to get a better image of average home value
# Found the county name for each fips id and now want to look at the distribution of the fips home values

def sqr_ft(df):
    sns.regplot(x='square_footage', y='tax_value', data=df.sample(2000), line_kws={'color':'orange'})
    plt.xlabel('Square Feet')
    plt.ylabel('Home Value')
    plt.title('Property Value By Square Footage')
    plt.show()
# creating a regplot to show a regression line through a sample of 2000 to visualize relationship

def room_cnt(zil):
    zil['room_count'] = zil['bedrooms'] + zil['bathrooms']
    zil = zil.drop(columns = ['bedrooms', 'bathrooms'])
    return zil

def linear_regression(x, y):
    lm = LinearRegression()
    lm.fit(x, y)
    lm_preds = lm.predict(x)
    preds = pd.DataFrame({'actual':y,
                      'baseline':y.mean(),
                      'lm_preds':lm_preds})
    return preds

# Setting up linear regression

def assess(actual, predictions):
    return round(sqrt(mean_squared_error(actual, predictions)), 2)  

def lasso_lars(x, y):
    lasso = LassoLars(alpha = .1)
    lasso.fit(x, y)
    lasso_preds = lasso.predict(x)
    preds = pd.DataFrame({'actual':y,
                      'baseline':y.mean(),
                      'lasso_preds':lasso_preds})
    return preds
# Lasso Lars setup

def polynomial(x, y):
    pol = PolynomialFeatures(degree=2)
    pol.fit(x, y)
    poly = pol.transform(x)
    lin = LinearRegression()
    lin.fit(poly, y)
    poly_preds = lin.predict(poly)
    preds = pd.DataFrame({'actual':y,
                      'baseline':y.mean(),
                      'poly_preds':poly_preds})
    return preds
# running validate setup

