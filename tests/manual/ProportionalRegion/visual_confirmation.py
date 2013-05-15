import sys

from investigators.visuals import cv2
from investigators.visuals import TemplateFinder

# images are tough to test so it's nice to have a sanity check
# setup an image finder
template = cv2.imread('template 640x480.png')
mask = cv2.imread('template mask 640x480.png')
small, smaller = (240, 320), (120, 160)
imgf = TemplateFinder(template, sizes=(small, smaller), mask=mask,
                      acceptable_threshold=0.5,
                      immediate_threshold=0.1)

# display the stored templates just to see how "masking" works
cv2.imshow(str(small), imgf._templates[small])
cv2.imshow(str(smaller), imgf._templates[smaller])

# search for the template in a scene
scene = cv2.imread('scene with similar image at 320x240.png')
result = imgf.locate_in(scene)
if not result:
    print 'Could not find the template in the scene at the given sizes.'
    cv2.destroyAllWindows()
    sys.exit()

# highlight the discovered boundaries and size in the original image
top, left, bottom, right = result
height, width = bottom - top, right - left
cv2.rectangle(scene, (left, top), (right, bottom),
              (255, 0, 255), thickness=5)
cv2.putText(scene,
            'top, left, bottom, right: {}, {}, {}, {}'.format(top, left,
                                                              bottom, right),
            (0, 30), 0, 0.5, (255, 0, 255))
cv2.putText(scene, 'height x width: {} x {}'.format(height, width),
            (0, 60), 0, 0.5, (255, 0, 255))
cv2.imshow('discovered image', scene)
cv2.waitKey()
cv2.destroyAllWindows()
