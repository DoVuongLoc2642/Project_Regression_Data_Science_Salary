import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("eda_data.csv")

df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]

    #get dummy data
df_dum = pd.get_dummies(df_model)

    #Split train_test
X = df_dum.drop("avg_salary", axis =1)
y = df_dum["avg_salary"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Multiple linear regression
# X_sm = X = sm.add_constant(X)
# model = sm.OLS(y,X_sm)
# model.fit().summary()

    #Model1: LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)

#cross_val_score (Crossvalidation)
# - take sample from the validation set -> run the model on the sample -> evaluate on the validation set

cross_val_score(lm, x_train, y_train, scoring = 'neg_mean_absolute_error', cv=3) #negative MAE - # -20,33

    #Model2: Lasso regression
lm_l = Lasso(alpha=.13)
lm_l.fit(x_train, y_train)
np.mean(cross_val_score(lm_l, x_train, y_train, scoring = 'neg_mean_absolute_error', cv=3))  # -21,33

#Find alpha param for Lasso
alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100)) # alpha 0.1-1.0
    error.append(np.mean(cross_val_score(lml, x_train, y_train, scoring = 'neg_mean_absolute_error', cv=3)) )

plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns= ['alpha', 'error'])
print(df_err[df_err.error == max(df_err.error)])  # alpha = 0.13

    #Model3: Random Forest Regression
rf = RandomForestRegressor()
print(np.mean(cross_val_score(rf, x_train, y_train, scoring = 'neg_mean_absolute_error', cv=3)))

#GridSearch to find parameter

parameters = {
    'n_estimators':range(10,300,10),
    'criterion':('squared_error','absolute_error'),
    'max_features':('sqrt','log2')
}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(x_train,y_train)

print(gs.best_score_)
print(gs.best_estimator_)

# test ensembles
tpred_lm = lm.predict(x_test)
tpred_lml = lm_l.predict(x_test)
tpred_rf = gs.best_estimator_.predict(x_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)

#save model
import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump(pickl, open( 'model_file' + ".p", "wb" ) ) # save pickl into file model_file.p - 'wb': file được mở ở chế độ ghi nhị phân (write binary).

#load model
file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

a = model.predict(np.array(list(x_test.iloc[1,:])).reshape(1,-1))[0]

