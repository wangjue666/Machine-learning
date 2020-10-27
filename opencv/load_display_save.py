import cv2;
import matplotlib.pyplot as plt

image = cv2.imread("./imgs/1.webp")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape[1])
cv2.imwrite("./imgs/new.jpg", image)
plt.imshow(image)

plt.show()