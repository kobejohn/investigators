import numpy

try:
    import cv2
except ImportError:
    print('Couldn\'t find OpenCV (cv2) vision library.'
          'Trying to import fallback cv2 for windows.')
    from _cv2_win_fallback import cv2


class ImageFinder(object):
    def __init__(self, base_template, sizes=None, mask=None,
                 acceptable_threshold=0.5,
                 immediate_threshold=0.1):
        """Create an image finder.

        Arguments:
        base_template: valid image. See _standardize for details
        sizes: sequence of (height, width) tuples template will be resized to
        mask: gray image matched height and width to base_template
        thresholds (both): 0 to 1; lower is a harder threshold to match
        acceptable_threshold: return best match under this after all templates
        immediate_threshold: immediately return any match under this
        """
        standardized_img = self._standardize_img(base_template)
        if mask is None:
            standardized_mask = None
        else:
            standardized_mask = self._standardize_mask(mask)
        masked = self._mask(standardized_img, standardized_mask)
        self._templates = self._build_templates(masked, sizes)
        self._acceptable_threshold = acceptable_threshold
        self._immediate_threshold = immediate_threshold

    def locate_in(self, scene):
        """Return the location and size of the best match in the scene from the
         available internal templates.

        Arguments:
        scene_std: opencv (numpy) bgr, bgra or gray image.

        Return:
        tuple of location (top, left) and size (height, width)
        """
        scene_std = self._standardize_img(scene)
        minloc_minval_size = list()
        for size_h_w, template in self._templates.items():
            result = cv2.matchTemplate(scene_std, template,
                                       cv2.TM_SQDIFF)
            # for some reason TM_SQDIFF_NORMED does not behave well
            # so do a manual normalization to [0,1] range
            h, w = size_h_w
            # max difference^2 * 3 channels * image size
            # max_sqdiff = (255 ** 2) * 3 * h * w
            result /= numpy.max(result)
            min_val, max_val, min_left_top, max_left_top = cv2.minMaxLoc(result)
            if min_val < self._immediate_threshold:
                # return immediately if immediate better than imm. threshold
                return tuple(reversed(min_left_top)), size_h_w
            elif min_val < self._acceptable_threshold:
                minloc_minval_size.append((min_left_top, min_val, size_h_w))
        # if any acceptable matches found, then return the best one
        if minloc_minval_size:
            best_left_top, best_val, best_h_w = min(minloc_minval_size,
                                                    key=lambda lvs: lvs[1])
            best_top_left = tuple(reversed(best_left_top))
            return best_top_left, best_h_w
        return None

    # helper methods
    def _standardize_img(self, img):
        """Convert valid image to numpy bgr or raise TypeError for invalid."""
        # get the channels
        try:
            actual_channels = img.shape[2]
        except IndexError:
            actual_channels = None  # grayscale doesn't have the extra item
        except AttributeError:
            actual_channels = -1  # it's not an numpy image
        # try to convert to opencv BGR
        bgr = 3
        bgra = 4
        gray = None
        converters_and_args = {bgr: (lambda x: x.copy(), (img,)),  # copy only
                               bgra: (cv2.cvtColor, (img, cv2.COLOR_BGRA2BGR)),
                               gray: (cv2.cvtColor, (img, cv2.COLOR_GRAY2BGR))}
        try:
            conversion_method, args = converters_and_args[actual_channels]
        except KeyError:
            raise TypeError('Unexpected image type:\n{}'.format(img))
        return conversion_method(*args)

    def _standardize_mask(self, mask):
        """Convert valid mask to numpy single-channel and black/white."""
        # get the channels
        try:
            actual_channels = mask.shape[2]
        except IndexError:
            actual_channels = None  # grayscale doesn't have the extra item
        except AttributeError:
            actual_channels = -1  # it's not an numpy image
        # try to convert to opencv Gray
        bgr = 3
        bgra = 4
        gray = None
        convertor = {bgr: (cv2.cvtColor, (mask, cv2.COLOR_BGR2GRAY)),
                     bgra: (cv2.cvtColor, (mask, cv2.COLOR_BGRA2GRAY)),
                     gray: (lambda x: x.copy(), (mask,))}  # copy only
        try:
            conversion_method, args = convertor[actual_channels]
        except KeyError:
            raise TypeError('Unexpected mask type:\n{}'.format(mask))
        converted = conversion_method(*args)
        # threshold the mask to black/white
        nonzeros = numpy.nonzero(converted)
        converted[nonzeros] = 255  # max out the non-zeros
        return converted

    def _mask(self, img, mask):
        """Mask the given image by applying random noise according to the mask.

        Arguments:
        img: an opencv bgr image (numpy shape (h, w, 3)). will be modified
        mask: an opencv single-channel image (numpy shape (h, w))
              with 0 for every pixel to be masked in img

        Returns:
        The original img object is modified and also returned
        """
        if mask is None:
            return img  # passthrough if no mask
        # prepare a sequence of noise the same size as the masked area
        #   Credit to J.F. Sebastion on StackOverflow for the basis of the
        #   quick random noise generator:
        #   http://stackoverflow.com/a/5685025/377366
        zeros = numpy.where(mask == 0)
        amount_of_noise = len(zeros[0])  # x and y are in two arrays so pick one
        channels = 3
        noise = numpy.frombuffer(numpy.random.bytes(channels * amount_of_noise),
                                 dtype=numpy.uint8)
        noise = noise.reshape((-1, channels))
        # apply the noise to the masked positions
        img[zeros] = noise
        return img

    def _build_templates(self, image, sizes):
        """Make sized versions of the base image and store them in a dict."""
        sized_templates = dict()
        if not sizes:
            # if no sizes provided, use the image directly as the template
            sized_templates[image.shape[:2]] = image
        else:
            for size in sizes:
                cv2_size = tuple(reversed(size))
                resized = cv2.resize(image, cv2_size,
                                     interpolation=cv2.INTER_AREA)
                sized_templates[size] = resized
        return sized_templates


if __name__ == '__main__':
    pass