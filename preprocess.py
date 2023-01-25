# preprocess.py
import pandas   as pd

from sklearn                    import preprocessing
from sklearn.model_selection    import train_test_split

# Reading the dataset
df = pd.read_csv('data/heart_2020_cleaned.csv')

# Checking and Removing Duplicate rows
df = df.drop_duplicates()

#  Data Encoding
encode_AgeCategory = {'55-59':57, '80 or older':80, '65-69':67,
                      '75-79':77,'40-44':42,'70-74':72,'60-64':62,
                      '50-54':52,'45-49':47,'18-24':21,'35-39':37,
                      '30-34':32,'25-29':27}
df['AgeCategory'] = df['AgeCategory'].apply(lambda x: encode_AgeCategory[x])
df['AgeCategory'] = df['AgeCategory'].astype('float')

# Integer encode columns with 2 unique values
for col in ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']:
    if df[col].dtype == 'O':
        le = preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col])

# One-hot encode columns with more than 2 unique values
df = pd.get_dummies(df, columns=['Race', 'Diabetic', 'GenHealth', ], prefix = ['Race', 'Diabetic', 'GenHealth'])

# Feature Scaling
standardScaler = preprocessing.StandardScaler()
columns_to_scale = ['BMI', 'PhysicalHealth', 'MentalHealth', 'AgeCategory', 'SleepTime']
df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])

# Undersampling
Y = df['HeartDisease']
X = df.drop(['HeartDisease'], axis = 1)

# Separation of the data set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 2)

X_train.to_csv('data/X_train.csv', index = 0)
X_test.to_csv('data/X_test.csv', index = 0)
y_train.to_csv('data/y_train.csv', index = 0)
y_test.to_csv('data/y_test.csv', index = 0)