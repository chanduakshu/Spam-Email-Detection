# Spam Email Detection using Machine Learning
# Algorithm: Multinomial Naive Bayes
# Level: Beginner

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset (Spam = 1, Ham = 0)
emails = [
    "Win money now",
    "Congratulations you won a prize",
    "Free entry in a lottery",
    "Call now for free offer",
    "Hey, are we meeting tomorrow?",
    "Please review the attached document",
    "Let's have lunch today",
    "Project meeting at 10 AM",
    "You have won free cash",
    "Exclusive offer just for you"
]

labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]

# Convert text data into numerical form
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(emails)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# User input
print("\n📧 Spam Email Detection System")
user_input = input("Enter email text: ")

input_vector = vectorizer.transform([user_input])
prediction = model.predict(input_vector)

if prediction[0] == 1:
    print("Result: 🚫 Spam Email")
else:
    print("Result: ✅ Not Spam (Ham)")

