import pandas as pd
from datetime import datetime

df = pd.read_csv("glassdoor_jobs.csv")

    #Salary Estimate -

df["hourly"] = df["Salary Estimate"].apply(lambda x: 1 if "per hour" in x.lower() else 0)
df["employer_provided"] = df["Salary Estimate"].apply(lambda x: 1 if "employer provided salary:" in x.lower() else 0)

df = df[df["Salary Estimate"] != "-1"] # Delete missing value

salary = df["Salary Estimate"].apply(lambda x: x.split("(")[0])  #Lấy phần tử đầu tiên sau khi split (
minus_Kd = salary.apply(lambda x: x.replace("K", " ").replace("$", " "))  #Delete K,$
min_hr = minus_Kd.apply(lambda x: x.lower().replace("per hour", " ")).apply(lambda x: x.lower().replace("employer provided salary:", " "))

#min,max,avg salary
df["min_salary"] = min_hr.apply(lambda x: int(x.split("-")[0]))
df["max_salary"] = min_hr.apply(lambda x: int(x.split("-")[1]))
df["avg_salary"] = (df["max_salary"] + df["min_salary"]) / 2

    #Company Name
df["company_txt"] = df["Company Name"].apply(lambda x: x.split("\n")[0])

    #Location - get the state only.
df["job_state"] = df["Location"].apply(lambda x: "CA" if "Los Angeles" in x else x.split(", ")[1] )
df["same_state"] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

    #Age of company - Founded columns
df["age"] = df["Founded"].apply(lambda x: x if x < 0 else datetime.now().year - x)

    #Job Description
# python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

# r studio
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
df.R_yn.value_counts()

# spark
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df.spark.value_counts()

# aws
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df.aws.value_counts()

# excel
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df.excel.value_counts()

df_out = df.drop(["Unnamed: 0"], axis= 1)
df_out.to_csv("salary_data_cleaned.csv", index= False)