#%%
# importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyodbc


#%%
# creating related Folder for working effectively 

# saving path for creating folder
path_report = "D:\\Projects\\Unemployment Rate Investigation\\Report"
path_Tables = "D:\\Projects\\Unemployment Rate Investigation\\Tables and Graphs"

# create the folder if directories don't exist
if not os.path.exists(path_report):
    os.makedirs(path_report)

if not os.path.exists(path_Tables):
    os.makedirs(path_Tables)


# %%
# import our Data and save

# Replace the file path and driver with your specific Access database information
data_path = r"D:\\Projects\\Unemployment Rate Investigation\\Data and Code\\LFS_RawData99.mdb"
driver = "{Microsoft Access Driver (*.mdb, *.accdb)}"

# Establish a connection to the Access database
conn_str = f'DRIVER={driver};DBQ={data_path}'
connection = pyodbc.connect(conn_str)

# Create a cursor to interact with the database
cursor = connection.cursor()

lfs = "LFS_RawData"
query = f"SELECT * FROM {lfs}"
cursor.execute(query)

data = cursor.fetchall()
columns = [column[0] for column in cursor.description]

df = pd.read_sql(query , connection)
df.to_csv("LFS99.csv")

cursor.close()
connection.close()


# %%
df = pd.read_csv("LFS99.csv")

# %%
# Data Cleaning

# selecting specific Columns for our investigation
col = "pkey F2_D03 F2_D04 F2_D07 F2_D08 F2_D15 F2_D16 F2_D17 F2_D19 F3_D01 F3_D02 F3_D03 F3_D04 F3_D05 F3_D06 F3_D07 F3_D08 F3_D31 F3_D33 F3_D34 F3_D36 IW_Yearly"
df = df[col.split()]

# renaming columns
df = df.rename({"pkey" : "ID Number" , 
                "F2_D03" : "Relationship" ,
                "F2_D04" : "Gender"  , 
                "F2_D07" : "Age", 
                "F2_D08" : "Nationality",
                "F2_D15" : "Education Status" ,
                "F2_D16" : "Literacy" , 
                "F2_D17" : "Education" , 
                "F2_D19" : "Marital Status" , 
                "F3_D36" : "Activity Status" , 
                "IW_Yearly" : "Weight" , 
                "F3_D31" : "Question 31" , 
                "F3_D33" : "Question 33" , 
                "F3_D34" : "Question 34"} , axis="columns")

q1 = "Question"
f1 = "F3_D0"

q2 = [q1 + " " + str(i) for i in range(1, 9)]
f2 = [f1 + str(i) for i in range(1, 9)]

dic_ = dict(zip(f2, q2))
df = df.rename(columns=dic_)

#%%
# changing our Data
# converting our data to float and if there is empty cell it return nan
df = df.apply(pd.to_numeric, errors='coerce')

q2.append("Question 31")
coln = ["Gender" , "Education Status" , "Literacy" , "Nationality"]
coln = coln + q2
for i in coln:
    df.loc[df[i] == 2 , i] = 0

df.loc[df["Nationality"] == 3 , "Nationality"] = 2
df.loc[df["Marital Status"].isin([float(i) for i in range(2,5)]) , "Marital Status"] = 0
df.loc[df["Relationship"].isin([float(i) for i in range(4,12)]) , "Relationship"] = 0
df.loc[df["Activity Status"].isin([float(i) for i in range(2,7)]) , "Activity Status"] = 0

# %%
df.to_csv('Cleaned_Data.csv', index=False)
# %%
# Load data
lfs_data = pd.read_csv("Cleaned_Data.csv")

est_data = lfs_data.groupby(['age', 'gender', 'mari', 'edu', 'nationality', 'lit'])['weight'].sum()
est_data.describe()  # Summary statistics

#%%
plt.figure(figsize=(10, 6))
plt.hist(lfs_data['age'], weights=lfs_data['weight'], bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Percentage')
plt.title('Histogram of Age')
plt.savefig("Histogram_age.png")  # Save the histogram
plt.show()


lit_1_mean = lfs_data.loc[lfs_data['lit'] == 'Literate', 'age'].mean()
lit_0_mean = lfs_data.loc[lfs_data['lit'] == 'Illiterate', 'age'].mean()

lit_1_summary = lfs_data.loc[lfs_data['lit'] == 'Literate', 'age'].describe()
lit_0_summary = lfs_data.loc[lfs_data['lit'] == 'Illiterate', 'age'].describe()

print("Mean age for literate individuals:", lit_1_mean)
print("Mean age for illiterate individuals:", lit_0_mean)
print("Summary statistics for literate individuals:\n", lit_1_summary)
print("Summary statistics for illiterate individuals:\n", lit_0_summary)

#%%
nationality_summary = lfs_data.groupby('nationality')['lit'].mean()
nationality_summary.plot(kind='bar', color=['blue', 'green'])
plt.xlabel('Nationality')
plt.ylabel('Mean Literacy')
plt.title('Mean Literacy by Nationality')
plt.xticks(np.arange(3), ['Afqan', 'Irani', 'Other'], rotation=0)
plt.legend(['Mean Literacy'])
plt.savefig("Mean_Literacy_by_Nationality.png")  # Save the bar plot
plt.show()

#%%
lfs_data['agestu'] = (lfs_data['age'] >= 15).astype(int)
lfs_data['agestu'].replace({0: "out of LFS", 1: "In LFS range"}, inplace=True)

lfs_data = lfs_data[lfs_data['agestu'] == "In LFS range"].copy()

lfs_data['actstu'] = np.where((lfs_data['Q1'] == 1) | (lfs_data['Q2'] == 1) | (lfs_data['Q3'] == 1) |
                              (lfs_data['Q4'] == 1) | (lfs_data['Q5'] == 1) | (lfs_data['Q6'] == 1) |
                              (lfs_data['Q8'] == 1), 1, 0)

lfs_data.loc[((lfs_data['Q31'] == 1) & (lfs_data['Q34'] == 1)) | ((lfs_data['Q33'] == 1) & (lfs_data['Q34'] == 1)),
             'actstu'] = 0

actstu_tab = lfs_data['actstu'].value_counts(normalize=True, weights=lfs_data['weight']) * 100
print(actstu_tab)

#%%
lfs_data = lfs_data[(lfs_data['age'] >= 20) & (lfs_data['age'] <= 30)]
lfs_data.dropna(subset=['nationality', 'activity'], inplace=True)

plt.figure(figsize=(10, 6))
lfs_data.groupby(['nationality', 'activity'])['lit'].mean().unstack().plot(kind='bar', stacked=True)
plt.xlabel('Nationality')
plt.ylabel('Percentage')
plt.title('Employment and Literacy by Nationality')
plt.xticks(rotation=0)
plt.legend(['Employment', 'Literacy'])
plt.savefig("Employment_Literacy_Nationality.png")
plt.show()

#%%
est_data = lfs_data.groupby('activity').agg({'gender': ['sum', 'mean', 'std', 'min', 'max', 'count'],
                                             'mari': ['sum', 'mean', 'std', 'min', 'max', 'count'],
                                             'edu': ['sum', 'mean', 'std', 'min', 'max', 'count'],
                                             'age': ['sum', 'mean', 'std', 'min', 'max', 'count']})
est_data.columns = ['_'.join(col) for col in est_data.columns]
est_data.reset_index(inplace=True)
est_data.rename(columns={'activity': 'Activity'}, inplace=True)
est_data.to_latex("q6.tex", index=False)

#%%
lfs_data['groupAge'] = pd.cut(lfs_data['age'], bins=np.arange(15, 76, 5), labels=np.arange(1, 12))
lfs_data['groupAge'] = lfs_data['groupAge'].astype(int)

lfs_data['Unemployment'] = np.where(lfs_data['activity'] == "Unemployed", 1, 0)

collapse_data = lfs_data.groupby('groupAge').agg({'mari': 'mean', 'Unemployment': 'mean', 'edu': 'mean',
                                                  'weight': 'sum'}).reset_index()
collapse_data.rename(columns={'mari': 'rateMarrital', 'Unemployment': 'Unemployment_Rate', 'edu': 'MeanofEdu'},
                     inplace=True)

plt.figure(figsize=(10, 6))
plt.bar(collapse_data['groupAge'], collapse_data['MeanofEdu'], align='center', alpha=0.5)
plt.xlabel('Group Age')
plt.ylabel('Mean Education')
plt.title('Mean Education by Group Age')
plt.savefig("Mean_Education_Group_Age.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(collapse_data['groupAge'], collapse_data['rateMarrital'], align='center', alpha=0.5, label='Married Rate')
plt.bar(collapse_data['groupAge'], collapse_data['Unemployment_Rate'], align='center', alpha=0.5, label='Unemployment Rate')
plt.xlabel('Group Age')
plt.ylabel('Rate')
plt.title('Married and Unemployment Rate by Group Age')
plt.legend()
plt.savefig("Married_Unemployment_Rate_Group_Age.png")
plt.show()