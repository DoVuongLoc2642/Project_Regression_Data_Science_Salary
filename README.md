# Project_Regression_Data_Science_Salary
Collect, clean raw data from Internet and build models to predict data science average annual salary.

My model contain a tool that predict data science related job to help people have an understanding about this field and negotiate their income when it comes to interview.

First I scraped over 1000 job descriptions from glassdoor using python and selenium library then I do some Features Engineering with each columns of job description to determine the value companies put on diverse skills like python, excel, spark and aws. I also using 3 different models: Linear, Lasso and Random Forest Regressors using GridSearchCV to reach the best paramater for the model. Finally, I built a client facing API using flask to simulate interaction between a client and server in real-time.

# Installation


* Create a conda virtual environment and activate it:

```
conda create -n timesformer python=3.7 -y
source activate timesformer
```
* Then, install the following packages: pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle

You can also install all the requirements packakes: ```pip install -r requirements.txt```  

