import numpy

try:
    import cv2
except ImportError:
    print('Couldn\'t find OpenCV (cv2) vision library.'
          'Trying to import fallback cv2 for windows.')
    from _cv2_win_fallback import cv2


class TemplateFinder(object):
    def __init__(self, template, sizes=None, mask=None,
                 acceptable_threshold=0.5,
                 immediate_threshold=0.1):
        """
        Arguments:
        - template: valid image. See _standardize for details
        - sizes: sequence of (height, width) tuples template will be resized to
        - mask: gray image matched height and width to template
        - acceptable_threshold: return best match under this after all templates
        - immediate_threshold: immediately return any match under this
            both thresholds: 0 to 1; lower is a harder threshold to match
        """
        template_std = self._standardize_image(template)
        if mask is None:
            mask_std = None
        else:
            mask_std = self._standardize_mask(mask)
        masked = self._mask(template_std, mask_std)
        self._templates = self._build_templates(masked, sizes)
        self.acceptable_threshold = acceptable_threshold
        self.immediate_threshold = immediate_threshold

    def locate(self, scene):
        """Return the boundaries and image of the best template/size
        match in the scene.

        Arguments:
        scene_std: opencv (numpy) bgr, bgra or gray image.

        Return:
        tuple of boundaries (top, left, bottom, right) and result image
        """
        scene_std = self._standardize_image(scene)
        scene_h, scene_w = scene_std.shape[0:2]
        matchvals_and_borders = list()
        for (template_h, template_w), template in self._templates.items():
            if (template_h > scene_h) or (template_w > scene_w):
                # skip if template too large. would cause ugly opencv error
                continue
            result = cv2.matchTemplate(scene_std, template, cv2.TM_SQDIFF)
            # for some reason TM_SQDIFF_NORMED does not behave well in tests
            # so do a manual normalization (avoiding zero division error)
            norm = numpy.max(result)
            if norm:  # i.e. don't normalize if it's all zeros (divide by zero)
                result /= norm
            min_val, max_val, (min_left, min_top), max_left_top\
                = cv2.minMaxLoc(result)
            top, left = min_top, min_left
            bottom, right = min_top + template_h, min_left + template_w
            if min_val < self.immediate_threshold:
                # return immediately if immediate better than imm. threshold
                return top, left, bottom, right
            elif min_val < self.acceptable_threshold:
                matchvals_and_borders.append((min_val,
                                              (top, left, bottom, right)))
        # if any acceptable matches found, then return the best one
        if matchvals_and_borders:
            match_val, (top, left, bottom, right) = min(matchvals_and_borders,
                                                        key=lambda x: x[0])
            return top, left, bottom, right
        # explicitly satisfy specification to return None when failed
        return None

    # helper methods
    def _standardize_image(self, img):
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

    def _mask(self, image, mask):
        """Mask the given image by applying random noise according to the mask.

        Arguments:
        image: an opencv bgr image (numpy shape (h, w, 3)). will be modified
        mask: an opencv single-channel image (numpy shape (h, w))
              with 0 for every pixel to be masked in image

        Returns:
        The original image object is modified and also returned
        """
        if mask is None:
            return image  # passthrough if no mask
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
        image[zeros] = noise
        return image

    def _build_templates(self, image, height_widths):
        """Make sized versions of the base image and store them in a dict.

        Size keys are (height, width) tuples
        """
        sized_templates = dict()
        if not height_widths:
            # if no height_widths provided, use the original image size
            h, w = image.shape[0:2]
            sized_templates[(h, w)] = image
        else:
            for h, w in height_widths:
                cv2_size = (w, h)  # reverse of numpy for cv2 coordinates
                resized = cv2.resize(image, cv2_size,
                                     interpolation=cv2.INTER_AREA)
                sized_templates[(h, w)] = resized
        return sized_templates


if __name__ == '__main__':
    pass