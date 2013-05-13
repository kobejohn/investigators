import random

import numpy

try:
    import cv2
except ImportError:
    print('Couldn\'t find OpenCV (cv2) vision library.'
          'Trying to import fallback cv2 for windows.')
    from _cv2_win_fallback import cv2


class ImageFinder(object):
    def __init__(self, base_template, sizes=None, mask=None):
        standardized = self._standardize(base_template)
        masked = self._mask(standardized, mask)
        self._base = masked
        self._templates = self._build_templates(self._base, sizes)

    # helper methods
    def _standardize(self, img):
        """Convert valid img to numpy bgr or raise TypeError for invalid."""
        # get the channels
        try:
            actual_channels = img.shape[2]
        except IndexError:
            actual_channels = None  # grayscale doesn't have the extra item
        except AttributeError:
            actual_channels = -1  # it's not an numpy img
        # try to convert to opencv BGR
        bgr = 3
        bgra = 4
        gray = None
        converters_and_args = {bgr: (lambda x: x, (img,)),  # passthrough
                               bgra: (cv2.cvtColor, (img, cv2.COLOR_BGRA2BGR)),
                               gray: (cv2.cvtColor, (img, cv2.COLOR_GRAY2BGR))}
        try:
            conversion_method, args = converters_and_args[actual_channels]
        except KeyError:
            raise TypeError('Unexpected img type:\n{}'.format(img))
        return conversion_method(*args)

    def _mask(self, img, mask):
        """Mask the given image by applying random noise according to the mask.

        Arguments:
        img: an opencv bgr image (numpy shape (h, w, 3))
        mask: an opencv single-channel image (numpy shape (h, w))
              with 0 for every pixel to be masked in img
        """
        if mask is None:
            return img  # passthrough if no mask
        #todo: below code works, but slowly
        img_copy = numpy.copy(img)
        positions_to_randomize = numpy.where(mask == 0)
        for p in zip(positions_to_randomize[0], positions_to_randomize[1]):
            img_copy[p] = random.randint(0, 255)
        return img_copy

    def _build_templates(self, image, sizes):
        """Make sized versions of the base image and store them in a dict."""
        sized_templates = dict()
        if not sizes:
            # if no sizes provided, use the image directly as the template
            sized_templates[image.shape[:2]] = image
        else:
            for size in sizes:
                resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                sized_templates[size] = resized
        return sized_templates


if __name__ == '__main__':
    pass