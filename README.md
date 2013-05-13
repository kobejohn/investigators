ImageFinder
========

Find one image within another.

More specifically, this is optimized to prepare a template at multiple
resolutions and then search a given scene for each of the sizes of the template.

I am using it specifcally to identify applications within a screenshot when
the application may be at various resolutions.

Status
======

Basics are working as below.

Command Line Usage
================

    # setup an image finder (this can be run with the files from tests/manual)
    img = cv2.imread('img.png')
    mask = cv2.imread('mask.png')
    small, smaller = (320, 240), (160, 120)
    imgf = ImageFinder(img, sizes=(small, smaller), mask=mask)
    # display the stored templates
    cv2.imshow(str(small), imgf._templates[small])
    cv2.imshow(str(smaller), imgf._templates[smaller])
    cv2.waitKey()
    cv2.destroyAllWindows()
