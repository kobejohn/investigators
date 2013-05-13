import cv2

from imagefinder import ImageFinder

# images are tough to test so it's nice to have a sanity check
# setup an image finder
img = cv2.imread('img.png')
mask = cv2.imread('mask.png')
small, smaller = (320, 240), (160, 120)
imgf = ImageFinder(img, sizes=(small, smaller), mask=mask)
# display the stored templates
cv2.imshow(str(small), imgf._templates[small])
cv2.imshow(str(smaller), imgf._templates[smaller])
cv2.waitKey()
cv2.destroyAllWindows()
