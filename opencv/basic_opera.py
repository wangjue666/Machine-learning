import cv2;
import matplotlib.pyplot as plt
def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()  


image = cv2.imread("./imgs/1.webp")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
(h,w,c) = image.shape
print(h, w, c)

(r,g,b) = image[0, 0]
print(image[0][0], r,g,b)

cX,cY = (int(w/2), int(h/2))
print(cX,cY)
t1 = image[0:cX, 0:cY]
image[0:30, 0:30] = (0, 255, 0)
show(t1)



