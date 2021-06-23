# Importing important libraries :
import pandas as pd         # data processing 
import csv
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv("C:/Users/HP/Desktop/Churn-Prediction/data/Churn modeling.csv")

# Removing the unused or irrelevant columns :
to_drop = ['CustomerId',
           'RowNumber',
          'Surname']
df.drop(to_drop, inplace = True, axis = 1)

# Renaming the column names :
new_name = {'HasCrCard':'HasCreditCard',
           'Exited':'Churn'}
df.rename(columns = new_name, inplace = True)

# One Hot Encoding
# Get one hot encoding of column Geography
geography_dummies = pd.get_dummies(df['Geography'], prefix='Geography')

# Get one hot encoding of column Gender
gender_dummies = pd.get_dummies(df['Gender'], prefix='Gender')

# Drop column Geography as it is now encoded
df.drop(['Geography'], inplace = True, axis = 1)
# Join the encoded df
df = pd.concat([df, pd.DataFrame(geography_dummies)], axis=1)

# Drop column Gender as it is now encoded
df.drop(['Gender'], inplace = True, axis = 1)
# Join the encoded df
df = pd.concat([df, pd.DataFrame(gender_dummies)], axis=1)

# Oversampling Technique
x = df.drop('Churn', axis=1)
y = df['Churn']
sm = SMOTE(random_state=0)
X_resampled, y_resampled = sm.fit_resample(x, y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state=42)

# Instantiate the model
rf = RandomForestClassifier(n_estimators=54, max_depth=20, random_state=42)

# Fit the model
rf.fit(X_train, y_train) 

# Make pickle file of our model
pickle.dump(rf, open("model.pkl", "wb"))
