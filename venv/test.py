from sklearn import datasets
data = datasets.load_wine()
print(data)
from sklearn.model_selection import train_test_split
training_data, testing_data, training_target, testing_target = \
train_test_split(data.data, data.target, test_size=0.4)


from sklearn.tree import DecisionTreeClassifier
my_tree = DecisionTreeClassifier(criterion='entropy', max_depth=10)
my_tree.fit(training_data, training_target)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(testing_target,my_tree.predict(testing_data)))
from sklearn.metrics import accuracy_score
print(accuracy_score(testing_target, my_tree.predict(testing_data)))