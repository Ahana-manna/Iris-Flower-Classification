from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data  
y = iris.target  # Target: species (setosa, versicolor, virginica)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict the species on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optionally, make predictions on new data
new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = clf.predict(new_sample)
print("Predicted species:", iris.target_names[prediction][0])
