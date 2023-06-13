import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
corr = np.corrcoef(X.T)
print(X)
print(corr)
sns.set(font_scale=1.2)
sns.heatmap(corr, annot=True, cmap='coolwarm', 
            xticklabels=iris.feature_names, yticklabels=iris.feature_names)
plt.title('Correlation Matrix for Iris Dataset')
plt.show()
