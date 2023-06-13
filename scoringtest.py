from sklearn.metrics import classification_report
y_pred = [0, 1, 2, 2]
y_true = [0, 1, 3, 2]
print(classification_report(y_pred, y_true))
# accuracy = accuracy_score(y_pred, y_true)
# print('Accuracy: %f' % accuracy)