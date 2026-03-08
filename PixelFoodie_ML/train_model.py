import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# 1. Loading updated data
df = pd.read_csv('recipes_data.csv')

# 2. X-la ippo 5 inputs irukku (Accuracy increase aagum)
X = df[['Spicy_Level', 'Prep_Time', 'Vegetarian', 'Meal_Type', 'Cuisine']]
y = df['Category']

# 3. Training the Model
model = DecisionTreeClassifier()
model.fit(X, y)

# 4. Saving the updated Model
with open('pixel_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Updated Model with 5 Inputs Trained Successfully!")