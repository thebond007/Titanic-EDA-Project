import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = sns.load_dataset("titanic")
print(df.head())
print(df.shape)

print(df.columns)

print(df.info())

print(df.describe())
'''How many rows?
How many columns?
Which columns are numeric?
Which columns contain missing values?'''

#print(df.isnull().sum())
df = df.drop(columns=['deck']) #drop the entire column from DataFrame
df['age'] = df['age'].fillna(df['age'].median()) #Fill all missing ages with the median age of the dataset
df = df.dropna(subset=['embarked', 'embark_town'])

#print(df.isnull().sum()) #drop those 2 specific rows entirely.

#Step 5: Univariate Analysis (Study one column at a time.)
#Survival count
sns.countplot(x='survived', data=df, hue='sex') #How many survived and How many died?
plt.title('Survival Count Broken Down by Gender')
plt.xlabel('Survived (0 = Died, 1 = Survived)')
plt.ylabel('Number of Passengers')
plt.savefig('survival_count.png')
plt.show()

#Age distribution
# dropna() temporarily ignores the missing values just for the plot
sns.histplot(data=df.dropna(subset=['age']), x='age', bins=20, hue='survived', multiple='stack', kde=True)
plt.title('Age Count Broken Down by Survived')
plt.xlabel('Most passengers belonged to which age group?')
plt.ylabel('Number of Passengers')
plt.savefig('age_distribution.png')
plt.show()

#Fare distribution (Are there outliers?)
sns.boxplot(data=df, x='survived', y='fare', hue='class')
#It will show you how fares looked for 1st, 2nd, and 3rd class passengers who lived versus those who died.
plt.title('How Ticket Fare Impact Survival')
plt.xlabel('Survvival Status (0 = Died, 1 = Survived)')
plt.ylabel('Fare Paid')
plt.savefig('fare_boxplot.png')
plt.show()

#Step 6: Bivariate Analysis (Study relation between two variables)
#Gender vs survival (Who survived more - Male or Female?)
sns.countplot(x='sex', hue='survived', data=df)
plt.title('Study relation between two variables')
plt.xlabel('Survival Gender (0 = Died, 1 = Survived)')
plt.ylabel('Passenger Count')
plt.savefig('Whosurvived_more_bygender.png')
plt.show()

#Passenger class vs survival
sns.countplot(x='class', hue='survived', data=df) #(Did first-class passengers survive more?)
plt.title('Survival Count Broken Down by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Passenger Count')
plt.savefig('Whosurvived_more_byclass.png')
plt.show()

#Age vs survival
sns.violinplot(x='survived', y='age', hue='sex', data=df, split=True) #(DId age affect survival)
plt.title("Age and Gender Distribution by Survival Status")
plt.xlabel("Survival Status (0 = Died, 1 = Survived)")
plt.ylabel('Passenger Age')
plt.savefig('did_age_affect_survival.png')
plt.show()

#Step 7: Multivariate Analysis
sns.pairplot(df, vars=['age', 'fare'], hue='survived')
#Are there patterns between age, fare and survival?
plt.suptitle("Multivariate Analysis: Age vs. Fare by Survival")
plt.savefig('pairplot.png')
plt.show()

#Step 8: Correlation heatmap
plt.figure(figsize=(8, 6))
corr = df.corr(numeric_only=True) #Which variables are strongly related?
sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5) 

plt.title("Correlation Heatmap of Titanic Variables", fontsize=14, pad=15)
plt.savefig('heatmap.png')

plt.show()










