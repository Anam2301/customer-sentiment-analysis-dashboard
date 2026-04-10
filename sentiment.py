import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned data
df = pd.read_csv("../data/cleaned_reviews.csv")

# Drop missing values (IMPORTANT FIX)
df = df.dropna(subset=["Review"])

# Features and target
X = df["Review"]
y = df["Label"]

# Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ==============================
# ✅ CREATE FINAL DATASET (IMPORTANT FOR ANALYST PART)
# ==============================

# Get original test reviews
df_test = df.iloc[y_test.index].copy()

# Add predictions
df_test["Predicted"] = y_pred

# Save file
df_test.to_csv("../data/final_output.csv", index=False)

print("✅ final_output.csv created!")

# ==============================
# ✅ OPTIONAL: LIVE PREDICTION
# ==============================

while True:
    text = input("\nEnter a review (or type 'exit'): ")

    if text.lower() == "exit":
        break

    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]

    if pred == 1:
        print("😊 Positive Review")
    else:
        print("😠 Negative Review")