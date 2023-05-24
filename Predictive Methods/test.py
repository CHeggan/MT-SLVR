import numpy as np
from sklearn import metrics




y_true = np.array([[1., 1., 1., 1., 0., 1., 1.],
        [0., 0., 0., 1., 0., 0., 0.]])

y_logits = [[0.7004, 0.5526, 0.5316, 0.4538, 0.4813, 0.5722, 0.5889],
        [0.4503, 0.5725, 0.5155, 0.4539, 0.3496, 0.6307, 0.5523]]


ap = metrics.average_precision_score(y_true, y_logits, average='macro')

print(ap)