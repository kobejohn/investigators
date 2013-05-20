from collections import namedtuple

import numpy

try:
    import cv2
except ImportError:
    print('Couldn\'t find OpenCV (cv2) vision library.'
          'Trying to import fallback cv2 for windows.')
    from _cv2_win_fallback import cv2


Rectangle = namedtuple('Rectangle', ('top', 'left', 'bottom', 'right'))
Dimensions = namedtuple('Dimensions', ('rows', 'columns'))


class ImageIdentifier(object):
    def __init__(self, templates,
                 acceptable_threshold=0.5, immediate_threshold=0.1):
        # convert the stored templates to standardized images
        templates_std = {k: _standardize_image(i) for k, i in templates.items()}
        self._templates_std = templates_std
        self.acceptable_threshold = acceptable_threshold
        self.immediate_threshold = immediate_threshold

    def identify(self, image):
        """Return the name of the best matching template or None if no match."""
        image_std = _standardize_image(image)
        acceptable_match_name_and_value = list()
        for name, template in self._templates_std.items():
            template_eq, image_eq = self._equalize(template, image_std)
            result = cv2.matchTemplate(image_eq, template_eq,
                                       method=cv2.TM_SQDIFF)
            # Instead of norming that stretches the image:
            # norm vs the worst case square difference:
            # (worst case one pixel)^2 * (TxI overlap)
            # 255^2 * T size
            norm = (255 ** 2) * template_eq.shape[0] * template_eq.shape[1]
            result /= float(norm)  # float just for paranoia
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            if minVal <= self.immediate_threshold:
                return name  # done!
            elif minVal <= self.acceptable_threshold:
                acceptable_match_name_and_value.append((name, minVal))
        best_name = None
        if acceptable_match_name_and_value:
            best_name, best_value = min(acceptable_match_name_and_value,
                                        key=lambda x: x[1])
        return best_name

    def _equalize(self, template, image):
        """Shrink template that doesn't fit or shrink image if template fits."""
        template_h, template_w = template.shape[0:2]
        image_h, image_w = image.shape[0:2]
        if (template_h <= image_h) and (template_w <= image_w):
            eq_template = template.copy()  # pass template through
            # template fits --> shrink image as little as possible
            h_scale = float(template_h) / image_h
            w_scale = float(template_w) / image_w
            scale = max(h_scale, w_scale)  # max --> minimum shrinking
            scaled_h = int(round(scale * image_h))
            scaled_w = int(round(scale * image_w))
            eq_image = cv2.resize(image, (scaled_w, scaled_h),
                                  interpolation=cv2.INTER_AREA)
        else:
            eq_image = image.copy()  # pass image through
            # template doesn't fit --> shrink template to completely fit
            h_scale = float(image_h) / template_h
            w_scale = float(image_w) / template_w
            scale = min(h_scale, w_scale)  # min --> most shrinking
            scaled_h = int(round(scale * template_h))
            scaled_w = int(round(scale * template_w))
            eq_template = cv2.resize(template, (scaled_w, scaled_h),
                                     interpolation=cv2.INTER_AREA)
        return eq_template, eq_image


class Grid(object):
    def __init__(self, dimensions, cell_padding):
        """Arguments:
        - dimensions: (rows, columns) for the grid. Each value is >= 1
        - cell_padding: (top, left, bottom, right) proportion to crop
            from each border
            Each value is [0, 1]
        """
        self.dimensions = dimensions
        self.cell_padding = cell_padding

    # Explicit property makes setter mocking easier (possible?) for testing
    def _get_dimensions(self):
        return self._dimensions

    def _set_dimensions(self, dimensions):
        _validate_dimensions(dimensions)
        self._dimensions = Dimensions(*dimensions)

    dimensions = property(_get_dimensions, _set_dimensions)

    # Explicit property makes setter mocking easier (possible?) for testing
    def _get_cell_padding(self):
        return self._cell_padding

    def _set_cell_padding(self, padding):
        # convert to relative window instead of the size from each border
        top, left, bottom, right = padding
        padded_cell = Rectangle(top, left, 1-bottom, 1-right)
        _validate_proportions(padded_cell)
        self._padded_cell = padded_cell
        self._cell_padding = padding  # this is not a rectangle so don't use it

    cell_padding = property(_get_cell_padding, _set_cell_padding)

    # main method
    def borders_by_grid_position(self, image):
        """Generate a sequence of tuples of:
            - relative grid positions (according to the grid dimensions)
            - pixel borders (according to the original image).
        """
        # pre-calculate step and padding values
        rows = self._dimensions.rows
        cols = self._dimensions.columns
        h, w = image.shape[0:2]
        # float steps
        row_step = float(h) / rows
        col_step = float(w) / cols
        # float padding
        padded_cell_top = row_step * self._padded_cell.top
        padded_cell_bottom = row_step * self._padded_cell.bottom
        padded_cell_left = col_step * self._padded_cell.left
        padded_cell_right = col_step * self._padded_cell.right
        for row in range(rows):
            for col in range(cols):
                top = int(round(row * row_step + padded_cell_top))
                left = int(round(col * col_step + padded_cell_left))
                bottom = int(round(row * row_step + padded_cell_bottom))
                right = int(round(col * col_step + padded_cell_right))
                grid_position = row, col
                cell_borders = Rectangle(top, left, bottom, right)
                yield grid_position, cell_borders


class ProportionalRegion(object):
    def __init__(self, rectangle_proportions):
        """Arguments:
        - rectangle_proportions: (top, left, bottom, right) proportions measured
            from (0, 0) at the top left corner of an image.
            Each value is [0, 1]
        """
        self.proportions = rectangle_proportions

    # Explicit property makes setter mocking easier (possible?) for testing
    def _get_proportions(self):
        return self._proportions

    def _set_proportions(self, rectangle_proportions):
        _validate_proportions(rectangle_proportions)
        self._proportions = Rectangle(*rectangle_proportions)

    proportions = property(_get_proportions, _set_proportions)

    # main method
    def region_in(self, image):
        """Return the border in pixels based on the stored proportions."""
        h, w = image.shape[0:2]
        top_proportion, left_proportion, bottom_proportion, right_proportion =\
            self.proportions
        top = int(round(h * top_proportion))
        left = int(round(w * left_proportion))
        bottom = int(round(h * bottom_proportion))
        right = int(round(w * right_proportion))
        return Rectangle(top, left, bottom, right)


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
        template_std = _standardize_image(template)
        if mask is None:
            mask_std = None
        else:
            mask_std = self._standardize_mask(mask)
        masked = self._mask(template_std, mask_std)
        self._templates = self._build_templates(masked, sizes)
        self.acceptable_threshold = acceptable_threshold
        self.immediate_threshold = immediate_threshold

    def locate_in(self, scene):
        """Return the boundaries and image of the best template/size
        match in the scene.

        Arguments:
        scene_std: opencv (numpy) bgr, bgra or gray image.

        Return:
        tuple of boundaries (top, left, bottom, right) and result image
        """
        scene_std = _standardize_image(scene)
        scene_h, scene_w = scene_std.shape[0:2]
        matchvals_and_borders = list()
        for (template_h, template_w), template in self._templates.items():
            if (template_h > scene_h) or (template_w > scene_w):
                # skip if template too large. would cause ugly opencv error
                continue
            result = cv2.matchTemplate(scene_std, template, cv2.TM_SQDIFF)
            # Instead of norming that stretches the image:
            # norm vs the worst case square difference:
            # (worst case one pixel)^2 * (TxI overlap)
            # 255^2 * T size
            norm = (255 ** 2) * template.shape[0] * template.shape[1]
            result /= float(norm)  # float just for paranoia
            min_val, max_val, (min_left, min_top), max_left_top\
                = cv2.minMaxLoc(result)
            bottom, right = min_top + template_h, min_left + template_w
            rectangle = Rectangle(min_top, min_left, bottom, right)
            if min_val < self.immediate_threshold:
                # return immediately if immediate better than imm. threshold
                return rectangle
            elif min_val < self.acceptable_threshold:
                matchvals_and_borders.append((min_val, rectangle))
        # if any acceptable matches found, then return the best one
        if matchvals_and_borders:
            match_val, rectangle = min(matchvals_and_borders,
                                       key=lambda x: x[0])
            return rectangle
        # explicitly satisfy specification to return None when failed
        return None

    # helper methods
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


# Helper functions
def _validate_proportions(rectangle_proportions):
    """Raise an error if the proportions will cause hard-to-trace errors."""
    # ValueError if out of bounds
    for border in rectangle_proportions:
        if (border < 0) or (1 < border):
            raise ValueError('Boundaries must be in the range [0, 1].')
    # ValueError if opposing borders are the same or reversed
    top, left, bottom, right = rectangle_proportions
    if (bottom <= top) or (right <= left):
        raise ValueError('There should be a positive gap between'
                         'both (bottom - top) and (right - left).')


def _validate_dimensions(dimensions):
    """Raise an error if the dimensions will cause hard-to-trace errors."""
    # ValueError if any dimension is < 1
    for dim in dimensions:
        if dim < 1:
            raise ValueError('Dimensions should be greater than or equal to 1.')


def _standardize_image(img):
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


if __name__ == '__main__':
    pass