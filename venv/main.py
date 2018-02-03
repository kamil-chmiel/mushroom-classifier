import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("venv/agaricus-lepiota.csv")

dataset = data.values[:, 1:-1]
target = data.values[:, 0]

le = preprocessing.LabelEncoder()
target = le.fit_transform(target)
for x in range(0, 21):
    dataset[:, x] = le.fit_transform(dataset[:, x])

training_data, testing_data, training_target, testing_target = \
    train_test_split(dataset, target.reshape(-1, 1), test_size=0.1)

my_tree = DecisionTreeClassifier(criterion='entropy', max_depth=10)
my_tree.fit(training_data, training_target)

print("Confusion Matrix:")
print(confusion_matrix(testing_target, my_tree.predict(testing_data)))
print("Accuracy score:")
print(accuracy_score(testing_target, my_tree.predict(testing_data)))