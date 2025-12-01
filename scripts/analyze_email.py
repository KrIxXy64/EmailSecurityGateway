import joblib

model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

email = input("Paste email text here:\n")

features = vectorizer.transform([email])
prediction = model.predict(features)[0]

print("\n--- RESULT ---")
print("Email is:", prediction.upper())
