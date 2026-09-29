import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


dataset = pd.read_csv("dataset/student_data.csv")
x = dataset[[ "hours_studied", "attendance"]]
y = dataset["passed"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

#the accuracy of the model on the test set
print(accuracy_score(y_test, predictions))
#score in the training and test 
train_accuracy = model.score(x_train, y_train)
test_accuracy = model.score(x_test, y_test)
print("Training accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)


#crossval
scores = cross_val_score(model, x, y, cv=5)

#print(scores)
#print(scores.mean())

#inaccuracy = confusion_matrix(y_test,predictions)
#print(inaccuracy)
#print(classification_report(y_test,predictions))
plt.boxplot([
    dataset[dataset["passed"] == 0]["hours_studied"],
    dataset[dataset["passed"] == 1]["hours_studied"]
])

plt.xticks([1, 2], ["Failed", "Passed"])
plt.title("Hours Studied by Result")
plt.xlabel("Result")
plt.ylabel("Hours Studied")
plt.show()