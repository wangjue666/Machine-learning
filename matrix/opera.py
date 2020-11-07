import numpy as np


A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = A
C = np.array([[1,2],[3,4],[5,6]])
D = np.array([[1],[2],[3]])
print(A, A.shape)


F = A - B

G = np.dot(A, B)
G2 = A*B
H = -A

print(G)

#矩阵转置
print(np.transpose(A))