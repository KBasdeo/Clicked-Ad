import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Predicts whether a user clicked on an ad based on the following variables:
# Daily Time Spent on Site, Age, Area Income, Daily Internet Usage, & Gender

desired_width = 410
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)

ad_data = pd.read_csv('advertising.csv')
print(ad_data.head())

# Checking for missing data
# sns.heatmap(ad_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# plt.show()

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

predictions = log_model.predict(X_test)

print(classification_report(y_test, predictions))
