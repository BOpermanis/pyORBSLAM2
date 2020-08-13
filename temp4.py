import numpy as np

coef = np.random.normal(0, 1, size=(4,))
pt = np.random.normal(0, 1, size=(3,))

coef[:3] = coef[:3] / np.linalg.norm(coef[:3])
print(coef)
print(pt)
exit()

e1 = np.asarray((- coef[2], 0, coef[0]))
e1 = e1 / np.linalg.norm(e1)

e2 = np.cross(coef[:3], e1)

A = np.asarray([

])