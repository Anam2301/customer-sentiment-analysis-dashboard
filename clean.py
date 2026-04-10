import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
df = pd.read_csv("../data/raw_reviews.csv")

# Lowercase
df["Review"] = df["Review"].str.lower()

# Remove punctuation
df["Review"] = df["Review"].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", str(x)))

# Remove stopwords
def remove_stopwords(text):
    words = text.split()
    filtered = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(filtered)

df["Review"] = df["Review"].apply(remove_stopwords)

print(" Stopwords Removed:")
print(df.head())
# Save cleaned dataset
df.to_csv("../data/cleaned_reviews.csv", index=False)

print(" Cleaned file saved as cleaned_reviews.csv")