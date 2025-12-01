import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("dataset/phishing_dataset.csv")

# Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email_text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model + vectorizer
joblib.dump(model, "models/phishing_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Training Completed. Model saved!")
