from collections import namedtuple

import ImageGrab
import numpy

try:
    import cv2
except ImportError:
    print('Couldn\'t find OpenCV (cv2) vision library.'
          'Trying to import fallback cv2 for windows.')
    from _cv2_win_fallback import cv2


Rectangle = namedtuple('Rectangle', ('top', 'left', 'bottom', 'right'))
Dimensions = namedtuple('Dimensions', ('rows', 'columns'))


def screen_shot():
    """Get a screenshot and return it as a standard image."""
    pil_img = ImageGrab.grab()
    numpy_img = _standardize_image(pil_img)
    return numpy_img


class TankLevel(object):
    def __init__(self, fill_bgr, empty_bgr, ignore_bgr):
        self.colors = fill_bgr, empty_bgr, ignore_bgr

    # explicit property methods to make mocking easy / possible
    def _get_colors(self):
        return self._colors

    def _set_colors(self, fill_empty_ignore):
        self._validate_colors(fill_empty_ignore)
        self._colors = fill_empty_ignore

    colors = property(_get_colors, _set_colors)

    def how_full(self, upright_tank_image):
        """Return an approximate proportion of how full the tank is from 0 to 1.

        Arguments:
        upright_tank_image: image of a tank, cropped as closely as possible
            and with filled part on the bottom of the image, empty part on top
        """
        tank_std = _standardize_image(upright_tank_image)
        fill, empty, ignore = self.colors
        # reduce palette
        reduced = self._reduce_palette(tank_std, fill, empty, ignore)
        # dilate to get rid of noise and allow good cropping
        dilated = self._proportional_dilate(reduced, fill, empty, ignore)
        # crop outer ignore region if ignore was specified
        if ignore is not None:
            cropped = self._crop(dilated, ignore)
        else:
            cropped = dilated
        # find longest horizontal line
        border_row = self._find_border_row(cropped, fill, empty)
        h = cropped.shape[0]
        fill_portion = float(h - border_row) / h
        return fill_portion

    def _reduce_palette(self, image, fill, empty, ignore):
        """Without changing data types, reduce the image to 2 or 3 colors."""
        base_shape = image.shape
        # allow room in these calculations for squares, negative, etc (int32)
        fill_img = numpy.ndarray(base_shape, dtype=numpy.int32)
        empty_img = numpy.ndarray(base_shape, dtype=numpy.int32)
        fill_img[::] = fill
        empty_img[::] = empty
        fill_dist =\
            numpy.sqrt(
                numpy.sum(
                    numpy.square(fill_img - image),
                    -1))
        empty_dist =\
            numpy.sqrt(
                numpy.sum(
                    numpy.square(empty_img - image),
                    -1))
        if ignore is not None:
            ignore_img = numpy.ndarray(base_shape, dtype=numpy.uint32)
            ignore_img[::] = ignore
            ignore_dist = \
                numpy.sqrt(
                    numpy.sum(
                        numpy.square(ignore_img - image),
                        -1))
            closest_to_fill = numpy.logical_and(fill_dist <= empty_dist,
                                                fill_dist <= ignore_dist)
            closest_to_empty = numpy.logical_and(empty_dist < fill_dist,
                                                 empty_dist <= ignore_dist)
            closest_to_ignore = numpy.logical_not(closest_to_fill +
                                                  closest_to_empty)
        else:
            closest_to_fill = fill_dist <= empty_dist
            closest_to_empty = numpy.logical_not(closest_to_fill)
        fill_points = numpy.nonzero(closest_to_fill)
        empty_points = numpy.nonzero(closest_to_empty)
        tank_reduced = numpy.ndarray(base_shape, dtype=numpy.uint8)
        tank_reduced.fill(0)
        tank_reduced[fill_points] = fill
        tank_reduced[empty_points] = empty
        if ignore is not None:
            ignore_points = numpy.nonzero(closest_to_ignore)
            tank_reduced[ignore_points] = ignore
        return tank_reduced

    def _proportional_dilate(self, image, fill, empty, ignore):
        """Dilate with fill > empty > ignore to get rid of noise lines."""
        # clearly differentiate and prioritize colors before dilating
        fill_positions = numpy.nonzero(numpy.all(image == fill, axis=-1))
        empty_positions = numpy.nonzero(numpy.all(image == empty, axis=-1))
        prioritized = numpy.ndarray(image.shape, dtype=numpy.uint8)
        prioritized_fill = 255
        prioritized_empty = 127
        prioritized[fill_positions] = prioritized_fill
        prioritized[empty_positions] = prioritized_empty
        if ignore is not None:
            ignore_positions =\
                numpy.nonzero(numpy.all(image == ignore, axis=-1))
            prioritized_ignore = 0
            prioritized[ignore_positions] = prioritized_ignore
        else:
            prioritized_ignore = None
        # dilate the prioritized colors
        h, w = image.shape[0:2]
        proportion = 0.1  # i.e. dilate with an element 5% of each dimension
        vertical = max(int(round(proportion * h)), 3)  # minimum size 3
        vertical += 0 if (vertical % 2) else 1  # make it odd if even
        horizontal = max(int(round(proportion * w)), 3)
        horizontal += 0 if (horizontal % 2) else 1  # make it odd if even
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (horizontal, vertical))
        dilated = cv2.dilate(prioritized, kernel)
        # convert any adjusted colors back to their closest palette color
        dilated = self._reduce_palette(dilated, prioritized_fill,
                                       prioritized_empty, prioritized_ignore)
        fill_positions =\
            numpy.nonzero(numpy.all(dilated == prioritized_fill, axis=-1))
        empty_positions =\
            numpy.nonzero(numpy.all(dilated == prioritized_empty, axis=-1))
        # replace the prioritized colors with the originals
        dilated[fill_positions] = fill
        dilated[empty_positions] = empty
        if ignore is not None:
            ignore_positions = \
                numpy.nonzero(numpy.all(dilated == prioritized_ignore, axis=-1))
            dilated[ignore_positions] = ignore
        return dilated

    def _crop(self, image, ignore):
        """Crop any borders that match ignore."""
        # thanks to Abid Rahman K on Stack Overflow for the nice crop technique
        # http://stackoverflow.com/a/13539194/377366
        ignore_map = numpy.all(image != ignore, axis=-1)
        non_ignore_positions = numpy.nonzero(ignore_map)
        ignore_or_not = numpy.ndarray(image.shape[0:2], dtype=numpy.uint8)
        ignore_or_not.fill(0)  # base is ignore
        ignore_or_not[non_ignore_positions] = 250
        _, thresh = cv2.threshold(ignore_or_not,
                                  245, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y + h, x: x + w]

    def _find_border_row(self, image, fill, empty):
        """Return the row of the following in order:
        - strongest horizontal border found in image
        - image top if fill color is prevalent
        - image bottom if empty color is prevalent
        - image bottom if same amount of fill and empty
        """
        edges = cv2.Canny(image, 80, 160)
        import math
        lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2)
        # return based on the strongest border:
        if lines is not None:
            fill_line = max(lines[0], key=lambda (x1, y1, x2, y2): abs(x2 - x1))
            return fill_line[1]
        # return based on balance of fill and empty color
        fill_count = len(numpy.nonzero(numpy.all(image == fill, axis=-1))[0])
        empty_count = len(numpy.nonzero(numpy.all(image == empty, axis=-1))[0])
        if fill_count > empty_count:
            return 0  # top of the image (full)
        # by default, return bottom of the image (empty)
        return image.shape[0] - 1  # bottom of the image

    def _validate_colors(self, fill_empty_ignore):
        # confirm 3 tuple
        try:
            fill, empty, ignore = fill_empty_ignore
        except (ValueError, TypeError):
            raise TypeError('Expected a tuple of three colors, but got {}'
                            ''.format(fill_empty_ignore))
        # confirm no 2 colors are the same
        if (fill == empty) or (fill == ignore) or (empty == ignore):
            raise ValueError('Expected the three colors to be different but'
                             ' got {}'.format(fill_empty_ignore))
        # confirm fill and empty are not None
        if (fill is None) or (empty is None):
            raise ValueError('Unexpectedly received None for fill or empty')


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
    def __init__(self, template, sizes=None, mask=None, scale_for_speed=1,
                 acceptable_threshold=0.5,
                 immediate_threshold=0.1):
        """
        Arguments:
        - template: valid image. See _standardize for details
        - sizes: sequence of (height, width) tuples template will be resized to
        - mask: valid matched height and width to template. 0 values are masked
        - scale_for_speed: (0, 1] optional template scale to speed up identify
        - acceptable_threshold: lowest threshold for identifying template
        - immediate_threshold: immediately threshold for identifying template
            both thresholds: [0, 1]; lower is a harder threshold to match
        """
        self.change_parts(template=template, mask=mask,
                          sizes=sizes, scale_for_speed=scale_for_speed)
        self.acceptable_threshold = acceptable_threshold
        self.immediate_threshold = immediate_threshold

    UNCHANGED = -987654

    def change_parts(self, template=UNCHANGED, mask=UNCHANGED,
                     sizes=UNCHANGED, scale_for_speed=UNCHANGED):
        """Change any parts provided and leave others the same before
        rebuilding the internal mechanism for identifying the template."""
        # template
        if not (template is self.UNCHANGED):  # leave it if unchanged
            self._template = _standardize_image(template)
        # mask - a little complicated since None has meaning
        if not (mask is self.UNCHANGED):  # leave it if unchanged
            if mask is None:  # mask has a special case of None
                self._mask = None
            else:
                self._mask = _standardize_image(mask)
        # sizes
        if not (sizes is self.UNCHANGED):  # leave it if unchanged
            self._validate_sizes(sizes)
            self._sizes = sizes
        # scale
        if not (scale_for_speed is self.UNCHANGED):  # leave it if unchanged
            self._validate_scale_for_speed(scale_for_speed)
            self._scale_for_speed = scale_for_speed or self._scale_for_speed
        # build new templates
        self._templates = self._build_templates()

    def _validate_sizes(self, sizes):
        """Raise an error unless None or sequence of 2-tuple numbers."""
        if sizes is None:
            return  # None is acceptable
        for h, w in sizes:
            int(round(h))
            int(round(w))

    def _validate_scale_for_speed(self, scale):
        """Raise an error if the scale seems to be invalid."""
        if (scale <= 0) or (1 < scale):
            raise ValueError('Unexpectedly received a speed scale factor'
                             ' outside of the range (0,1]: {}'.format(scale))

    def _build_templates(self):
        """Make and store sized versions of the base template.

        Keys are (height, width) size tuples based on original sizes
        Values are resized versions of the base template including speed scaling
        """
        # mask the base template
        masked = self._apply_mask(self.template, self.mask)
        # if no sizes, just use the size of the base template
        original_h, original_w = original_size = masked.shape[0:2]
        sizes = self.sizes or [(original_h, original_w)]
        sized_templates = dict()
        for base_h, base_w in sizes:
            base_size = base_h, base_w
            # resize the template based on the size and optional speed scaling
            final_h = int(round(base_h * self.scale_for_speed))
            final_w = int(round(base_w * self.scale_for_speed))
            final_size = (final_h, final_w)
            cv_final_size = final_w, final_h  # h/w are reversed for opencv
            if final_size == original_size:
                final_template = masked
            elif final_h > original_h:
                # cubic resize for enlargements
                final_template = cv2.resize(masked, cv_final_size,
                                            interpolation=cv2.INTER_CUBIC)
            else:
                # area interpolation for shrinking
                final_template = cv2.resize(masked, cv_final_size,
                                            interpolation=cv2.INTER_AREA)
            sized_templates[base_size] = final_template
        return sized_templates

    def _apply_mask(self, image, mask):
        """Mask the given image by applying random noise according to the mask.

        Arguments:
        image: an opencv bgr image (numpy shape (h, w, 3)). will be modified
        mask: an opencv bgr image with (0, 0, 0) for every pixel to mask

        Returns:
        New, masked image.
        """
        new_image = image.copy()
        if mask is None:
            return new_image  # just copy if no mask
        # prepare a noise image to be copied from
        zeros = numpy.all(mask == 0, axis=-1)
        # if amount_of_noise:
        h, w, channels = new_image.shape
        #   Credit to J.F. Sebastion on StackOverflow for the basis of the
        #   quick random noise generator:
        #   http://stackoverflow.com/a/5685025/377366
        noise = numpy.frombuffer(numpy.random.bytes(h * w * channels),
                                 dtype=numpy.uint8)
        noise = noise.reshape(new_image.shape)
        # apply the noise to the masked positions
        new_image[zeros] = noise[zeros]
        return new_image

    def locate_in(self, scene):
        """Return the boundaries and image of the best template/size
        match in the scene.

        Arguments:
        scene_std: pil rgb or opencv (numpy) bgr, bgra or gray image.

        Return:
        tuple of boundaries (top, left, bottom, right) and result image
        """
        # prepare the speed scaled and original scenes carefully
        scene_std = _standardize_image(scene)
        scene_original_h = scene_std.shape[0]
        scene_original_w = scene_std.shape[1]
        speed_scale = self.scale_for_speed
        if speed_scale == 1:
            scene_scaled = scene_std
        else:
            scene_scaled_h = int(round(scene_original_h * speed_scale))
            scene_scaled_w = int(round(scene_original_w * speed_scale))
            scene_scaled = cv2.resize(scene_std,
                                      (scene_scaled_w, scene_scaled_h),
                                      interpolation=cv2.INTER_AREA)
        matchvals_and_borders = list()
        HEIGHT = 1
        # test templates from smallest to largest to cover more earlier
        small_to_big_templates = sorted(self._templates.items(),
                                        key=lambda (size, templ): size[HEIGHT])
        for template_original_size, template in small_to_big_templates:
            template_original_h, template_original_w = template_original_size
            if (template_original_h > scene_original_h
                    or template_original_w > scene_original_w):
                # skip if template too large. would cause ugly opencv error
                continue
            # scale the template
            if speed_scale == 1:
                template_scaled_h, template_scaled_w =\
                    template_original_h, template_original_w
                template_scaled = template
            else:
                template_scaled_h = int(round(template_original_h
                                              * speed_scale))
                template_scaled_w = int(round(template_original_w
                                              * speed_scale))
                template_scaled = cv2.resize(template,
                                             (template_scaled_w,
                                              template_scaled_h),
                                             interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(scene_scaled, template_scaled,
                                       cv2.TM_SQDIFF)
            # Instead of norming that stretches the correlation map:
            # norm vs the worst case square difference:
            # (worst case one pixel)^2 * (TxI overlap)
            # 255^2 * T size
            norm = (255 ** 2) * template_scaled_h * template_scaled_w
            result /= float(norm)  # float just for paranoia
            min_val, max_val, (min_left, min_top), max_left_top\
                = cv2.minMaxLoc(result)
            # reverse the speed scaling to produce a rectangle in the original
            top_original = int(round(float(min_top) / speed_scale))
            left_original = int(round(float(min_left) / speed_scale))
            bottom_original = int(round(float(min_top + template_scaled_h)
                                        / speed_scale))
            right_original = int(round(float(min_left + template_scaled_w)
                                       / speed_scale))
            rectangle = Rectangle(top_original, left_original,
                                  bottom_original, right_original)
            if min_val < self.immediate_threshold:
                # return immediately if immediate better than imm. threshold
                return rectangle
            elif min_val < self.acceptable_threshold:
                # store this result for later if it's acceptable
                matchvals_and_borders.append((min_val, rectangle))
            else:
                pass  # ignore this result if it's not within acceptable level
        # after all templates, return the best match, if any
        if matchvals_and_borders:
            match_val, rectangle = min(matchvals_and_borders,
                                       key=lambda x: x[0])
            return rectangle
        # explicitly satisfy specification to return None when failed to locate
        return None

    @property
    def template(self):
        return self._template

    @property
    def mask(self):
        return self._mask

    @property
    def sizes(self):
        return self._sizes

    @property
    def scale_for_speed(self):
        return self._scale_for_speed


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
    UNKNOWN = None
    RGB = -3
    BGR = 3
    BGRA = 4
    GRAY = 1
    KNOWN_TYPES = (RGB, BGR, BGRA, GRAY)
    channels = UNKNOWN
    # basically, convert explicitly handled cases and throw an error otherwise.
    # Discover channels for PIL and conver to basic numpy
    if channels is UNKNOWN:
        try:
            channels = -1 * len(img.getbands())  # assume Gray or RGB
            img = numpy.asarray(img)
        except AttributeError:
            pass  # it wasn't PIL
    # Discover channels for Numpy
    if channels is UNKNOWN:
        try:
            channels = img.shape[2]  # BGR or BGRA or unsupported
        except IndexError:
            channels = GRAY  # numpy gray has no third shape element
        except AttributeError:
            pass  # it wasn't numpy
    # error if not an explicitly handled type
    if channels not in KNOWN_TYPES:
        raise TypeError('Unexpected image type:\n{}'.format(img))
    # Standardize the basic numpy image to BGR
    converters_and_args = {BGR: (lambda x: x.copy(), (img,)),  # copy only
                           RGB: (cv2.cvtColor, (img, cv2.COLOR_RGB2BGR)),
                           BGRA: (cv2.cvtColor, (img, cv2.COLOR_BGRA2BGR)),
                           GRAY: (cv2.cvtColor, (img, cv2.COLOR_GRAY2BGR))}
    conversion_method, args = converters_and_args[channels]
    return conversion_method(*args)


if __name__ == '__main__':
    pass