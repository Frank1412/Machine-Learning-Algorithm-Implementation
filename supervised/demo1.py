# -*-coding=utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.random.randn(6, 5)
print(x)
y = np.array([0, 1, 1, 0, 1, 1])

lr = LogisticRegression(fit_intercept=True, max_iter=100)
lr.fit(x,y)
pred = lr.predict_proba(x)
print(pred)