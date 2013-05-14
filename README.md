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

    import sys

    from imagefinder import cv2  # replace this with your cv2 if you have it
    from imagefinder import ImageFinder

    # images are tough to test so it's nice to have a sanity check
    # setup an image finder
    template = cv2.imread('template 640x480.png')
    mask = cv2.imread('template mask 640x480.png')
    small, smaller = (240, 320), (120, 160)
    imgf = ImageFinder(template, sizes=(small, smaller), mask=mask,
                       acceptable_threshold=0.5,
                       immediate_threshold=0.1)

    # display the stored templates just to see how "masking" works
    cv2.imshow(str(small), imgf._templates[small])
    cv2.imshow(str(smaller), imgf._templates[smaller])
    cv2.waitKey()
    cv2.destroyAllWindows()

    # search for the template in a scene
    scene = cv2.imread('scene with similar image at 320x240.png')
    result = imgf.locate_in(scene)
    if not result:
        print 'Could not find the template in the scene at the given sizes.'
        sys.exit()
    (top, left), (height, width) = result

    # highlight the disocovered location and size in the original image
    cv2.rectangle(scene, (left, top), (left + width, top + height),
                  (255, 0, 255), thickness=5)
    cv2.putText(scene, 'top left = ({}, {})'.format(top, left),
                (0, 30), 0, 1, (255, 0, 255))
    cv2.putText(scene, 'height width = ({} x {})'.format(height, width),
                (0, 60), 0, 1, (255, 0, 255))
    cv2.imshow('discovered image', scene)
    cv2.waitKey()
    cv2.destroyAllWindows()
