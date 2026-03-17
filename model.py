# 🌾 Crop Recommendation - Model Training & Saving

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("Crop_recommendation.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# ==============================
# 2. Prepare Data
# ==============================
X = df.drop("label", axis=1)
y = df["label"]

# ==============================
# 3. Train Model (Decision Tree)
# ==============================
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print("Model Trained Successfully!")

# ==============================
# 4. Save Model
# ==============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl ✅")

# ==============================
# 5. Test Prediction (Optional)
# ==============================
sample = pd.DataFrame(
    [[90, 40, 40, 25, 80, 6.5, 200]],
    columns=X.columns
)

prediction = model.predict(sample)
print("Sample Prediction:", prediction[0])