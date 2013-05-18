import unittest

from mock import patch
import numpy

from investigators import visuals
from investigators.visuals import cv2
from investigators.visuals import ProportionalRegion, TemplateFinder, Grid


class Test_Grid(unittest.TestCase):
    # Initialization
    def test___init___sets_grid_dimensions(self):
        some_dimensions = _generic_dimensions()
        g = self._generic_grid(dimensions=some_dimensions)
        self.assertEqual(g.dimensions, some_dimensions)

    def test___init___sets_cell_padding(self):
        some_proportions = _generic_proportions()
        g = self._generic_grid(cell_padding=some_proportions)
        self.assertEqual(g.cell_padding, some_proportions)

    # Configuration
    def test_setting_dimensions_validates_them(self):
        some_dimensions = (5, 7)
        other_dimensions = (11, 13)
        g = self._generic_grid(dimensions=some_dimensions)
        with patch.object(visuals, '_validate_dimensions') as m_validate:
            g.dimensions = other_dimensions
        m_validate.assert_called_with(other_dimensions)

    def test_seting_cell_padding_validates_the_proportions(self):
        some_proportions = (0, .1, .2, .3)
        other_proportions = (.4, .5, .6, .7)
        g = self._generic_grid(cell_padding=some_proportions)
        with patch.object(visuals, '_validate_proportions') as m_validate:
            g.cell_padding = other_proportions
        m_validate.assert_called_with(other_proportions)

    # Splitting an image into a grid
    def test_gridify_generates_correct_sequence_of_borders(self):
        # get the test image which is constructed as follows:
        from os import path
        this_path = path.abspath(path.split(__file__)[0])
        grid_image_path = path.join(this_path,
                                    'grid (tlbr padding - 1, 2, 3, 4).png')
        grid_image = cv2.imread(grid_image_path)
        self.assertIsNotNone(grid_image)  # just confirm file loaded
        # 4 rows, 2 columns
        dimensions = (4, 2)
        # (top, left, bottom, right) padding:
        #   = (1px, 2px, 3px, 4px)
        #   = (1/5, 2/10, 3/5, 4/10)
        #   = (0.2, 0.2, 0.6, 0.4)
        cell_padding = 0.2, 0.2, 0.6, 0.4
        #   ==> remaining cell image shape is 1 row, 4 columns (3 channels)
        channels = 3
        cell_shape_spec = (1, 4, channels)  # 1 row, 4 columns, 3 channels
        # ==> content of each cell image should be exactly the colored parts
        #     in the image as follows:
        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        white = (255, 255, 255)
        colors_spec = [[red, white],
                       [green, red],
                       [blue, green],
                       [white, blue]]
        colors_spec = numpy.asarray(colors_spec)
        # gridify the image and confirm all sub images match the above specs
        grid = Grid(dimensions, cell_padding)
        for grid_position, borders in grid.borders_by_grid_position(grid_image):
            top, left, bottom, right = borders
            # confirm the size
            cell_shape = (bottom - top, right - left, channels)
            self.assertEqual(cell_shape, cell_shape_spec)
            spec_pixel = colors_spec[grid_position]
            # confirm the color for each pixel in the cell
            for pixel_row in range(top, bottom):
                for pixel_col in range(left, right):
                    p = pixel_row, pixel_col
                    image_pixel = grid_image[p]
                    self.assertTrue(numpy.all(image_pixel == spec_pixel),
                                    'Pixel in grid cell:'
                                    '\n{}: {}'
                                    '\nunexpectedly not equal to specification:'
                                    '\n{}'.format(grid_position, image_pixel,
                                                  spec_pixel))

    def _generic_grid(self, dimensions=None, cell_padding=None):
        dimensions = dimensions or (5, 7)
        cell_padding = cell_padding or (0.1, 0.3, 0.5, 0.7)
        return Grid(dimensions, cell_padding)


class Test_ProportionalRegion(unittest.TestCase):
    # Initialization
    def test___init___sets_proportions(self):
        some_proportions = _generic_proportions()
        pr = ProportionalRegion(some_proportions)
        self.assertEqual(pr.proportions, some_proportions)

    # Configuration
    def test_seting_proportions_validates_them(self):
        some_proportions = (0, .1, .2, .3)
        other_proportions = (.4, .5, .6, .7)
        pr = ProportionalRegion(some_proportions)
        with patch.object(visuals, '_validate_proportions') as m_validate:
            pr.proportions = other_proportions
        m_validate.assert_called_with(other_proportions)

    # Returning the window
    def test_region_in_returns_correct_borders(self):
        # create an image to work with
        h, w = 100, 200
        image = _generic_image(height=h, width=w)
        # specify the correct values for a given proportion set
        top_proportion, left_proportion, bottom_proportion, right_proportion =\
            proportions =\
            _generic_proportions()
        top_spec = int(round(h * top_proportion))
        left_spec = int(round(w * left_proportion))
        bottom_spec = int(round(h * bottom_proportion))
        right_spec = int(round(w * right_proportion))
        border_pixels_spec = top_spec, left_spec, bottom_spec, right_spec
        # confirm the calculation
        pr = ProportionalRegion(proportions)
        border_pixels = pr.region_in(image)
        self.assertEqual(border_pixels, border_pixels_spec)


class Test_TemplateFinder(unittest.TestCase):
    # Initialization
    def test___init___standardizes_image(self):
        img = _generic_image()
        mask = _generic_image()
        with patch.object(TemplateFinder, '_standardize_image') as m_stdize:
            m_stdize.return_value = _generic_image(channels=3)
            TemplateFinder(img, mask=mask)
        m_stdize.assert_called_with(img)

    def test___init___standardizes_mask(self):
        img = _generic_image()
        mask = _generic_image()
        with patch.object(TemplateFinder, '_standardize_mask') as m_stdize:
            m_stdize.return_value = _generic_image(channels=None)
            TemplateFinder(img, mask=mask)
        m_stdize.assert_called_with(mask)

    def test___init___stores_original_size_if_no_sizes_provided(self):
        h, w = size_key = 20, 10
        img = _generic_image(height=h, width=w)
        imgf = TemplateFinder(img)
        self.assertTrue(size_key in imgf._templates)

    def test___init___applies_an_optional_mask_to_image(self):
        img = _generic_image()
        mask = _generic_image()
        with patch.object(TemplateFinder, '_mask') as m_mask:
            m_mask.return_value = _generic_image(channels=3)
            TemplateFinder(img, mask=mask)
        m_mask.assert_called()

    # Internal templates
    def test_internal_template_dict_keys_are_int_tuples_of_height_width(self):
        height_spec, width_spec = 10, 20
        imgf = self._generic_ImageFinder(height=height_spec, width=width_spec,
                                         sizes=None)
        for height, width in imgf._templates.keys():
            self.assertIs(height, height_spec)
            self.assertIs(width, width_spec)

    def test_internal_template_dict_has_a_bgr_template_per_size(self):
        imgf = self._generic_ImageFinder()
        bgr_channels = 3
        for (rows, cols), template in imgf._templates.items():
            size_spec = (rows, cols, bgr_channels)
            self.assertSequenceEqual(template.shape, size_spec)

    # Locating templates in a scene:
    def test_locate_in_standardizes_the_scene_image(self):
        # setup the image finder
        template = _generic_image(height=2, width=2)
        imgf = TemplateFinder(template, sizes=None)
        # confirm standardize is called when analyzing a scene
        scene = _generic_image(height=100, width=100)
        with patch.object(TemplateFinder, '_standardize_image') as m_stdize:
            m_stdize.return_value = _generic_image(channels=3)
            imgf._standardize_image(scene)
        m_stdize.assert_called_with(scene)

    def test_locate_returns_None_if_no_internal_templates_found_in_scene(self):
        # setup a template and scene that should be guaranteed not to be matched
        black_template = _generic_image(height=5, width=10)
        black_template.fill(0)
        white_scene = _generic_image(height=20, width=20)
        white_scene.fill(255)
        imgf = TemplateFinder(black_template)
        p = imgf.locate_in(white_scene)
        self.assertIsNone(p)

    def test_locate_returns_result_at_end_when_immediate_not_passed(self):
        top_spec, left_spec, bottom_spec, right_spec = 2, 3, 22, 33
        height_spec = bottom_spec - top_spec
        width_spec = right_spec - left_spec
        black_template = _generic_image(height=height_spec,
                                        width=width_spec)
        black_template.fill(0)
        white_scene = _generic_image(height=height_spec * 3,
                                     width=width_spec * 3)
        white_scene.fill(255)
        # set a black square to match the template
        white_scene[top_spec:bottom_spec, left_spec:right_spec] = 0
        # setup thresholds to exercise the spec
        impossible = -1
        always = 2
        imgf = TemplateFinder(black_template,
                              acceptable_threshold=always,
                              immediate_threshold=impossible)
        borders = imgf.locate_in(white_scene)
        self.assertEqual(borders,
                         (top_spec, left_spec, bottom_spec, right_spec))

    def test_locate_returns_result_immediately_when_immediate_passes(self):
        top_spec, left_spec, bottom_spec, right_spec = 2, 3, 22, 33
        height_spec = bottom_spec - top_spec
        width_spec = right_spec - left_spec
        black_template = _generic_image(height=height_spec,
                                        width=width_spec)
        black_template.fill(0)
        white_scene = _generic_image(height=height_spec * 3,
                                     width=width_spec * 3)
        white_scene.fill(255)
        # set a black square to match the template
        white_scene[top_spec:bottom_spec, left_spec:right_spec] = 0
        # setup thresholds to exercise the spec
        impossible = -1
        always = 2
        imgf = TemplateFinder(black_template,
                              acceptable_threshold=impossible,
                              immediate_threshold=always)
        # confirm the result
        borders = imgf.locate_in(white_scene)
        self.assertEqual(borders,
                         (top_spec, left_spec, bottom_spec, right_spec))

    def test_locate_in_ignores_templates_too_big_for_the_scene(self):
        # setup an image finder with only a large size for template
        large_h, large_w = 100, 200
        imgf = self._generic_ImageFinder(sizes=((large_h, large_w),))
        # setup a scene smaller than the large size template
        small_h, small_w = large_h - 20, large_w - 20
        small_scene = _generic_image(height=small_h, width=small_w)
        # confirm that matchTemplate is not called
        with patch.object(cv2, 'matchTemplate') as m_match:
            m_match.return_value = _generic_image(channels=None)
            imgf.locate_in(small_scene)
        self.assertFalse(m_match.called)

    # Internal specifications
    def test__standardize_img_mask_raise_TypeError_unless_bgr_bgra_or_gry(self):
        bgr = _generic_image(channels=3)
        bgra = _generic_image(channels=4)
        gray = _generic_image(channels=None)
        unhandled_channel_count = _generic_image(channels=2)
        just_a_string = 'whut'
        # confirm the bad one fails
        imgf = self._generic_ImageFinder()
        for standardizer in (imgf._standardize_image, imgf._standardize_mask):
            self.assertRaises(TypeError, standardizer, unhandled_channel_count)
            self.assertRaises(TypeError, standardizer, just_a_string)
            # confirm the valid ones don't fail
            for ok_img in (bgr, bgra, gray):
                standardizer(ok_img)

    def test__mask_puts_noise_on_masked_locations(self):
        h, w = 20, 10
        # make a special white image with values over random range
        high_value = 1000
        white = numpy.ndarray((h, w, 3), dtype=numpy.uint16)
        white.fill(high_value)
        # make a mask with zeros at specified positions
        mask = numpy.ndarray((h, w), dtype=numpy.uint8)
        mask.fill(255)
        masked_positions = [(0, 0), (5, 5)]
        for p in masked_positions:
            mask[p] = 0
        # use a generic image finder to test _mask
        imgf = self._generic_ImageFinder()
        masked_white = imgf._mask(white, mask)
        for p in masked_positions:
            masked_pixel = masked_white[p]
            # confirm all noise pixels have been changed
            self.assertTrue(numpy.all(masked_pixel != high_value))

    def _generic_ImageFinder(self, height=None, width=None,
                             channels=None, sizes=None):
        img = _generic_image(height=height, width=width, channels=channels)
        sizes = sizes
        return TemplateFinder(img, sizes=sizes)


class Test_Helpers(unittest.TestCase):
    def test__validate_proportions_raises_ValueError_if_opp_borders_rvrsd(self):
        v = 0.5
        left_right_same = _generic_proportions(left=v, right=v)
        top_bottom_same = _generic_proportions(top=v, bottom=v)
        self.assertRaises(ValueError,
                          visuals._validate_proportions, left_right_same)
        self.assertRaises(ValueError,
                          visuals._validate_proportions, top_bottom_same)

    def test__validate_proportions_raises_ValueError_if_borders_OOB(self):
        under_bound = -0.1
        over_bound = 1.1
        top_out = _generic_proportions(top=under_bound)
        left_out = _generic_proportions(left=under_bound)
        bottom_out = _generic_proportions(bottom=over_bound)
        right_out = _generic_proportions(right=over_bound)
        for out_of_bounds in (top_out, left_out, bottom_out, right_out):
            self.assertRaises(ValueError,
                              visuals._validate_proportions, out_of_bounds)

    def test__validate_dimensions_raises_ValueError_if_dim_less_than_one(self):
        half = 0.5, 3
        zero = 3, 0
        negative = -1, 3
        self.assertRaises(ValueError, visuals._validate_dimensions, half)
        self.assertRaises(ValueError, visuals._validate_dimensions, zero)
        self.assertRaises(ValueError, visuals._validate_dimensions, negative)


# Helper factories
def _generic_image(height=None, width=None, channels=None):
    height = height if not height is None else 30
    width = width if not width is None else 40
    if channels:
        shape = (height, width, channels)
    else:
        shape = (height, width)
    return numpy.zeros(shape, dtype=numpy.uint8)


def _generic_proportions(top=None, left=None, bottom=None, right=None):
    top = top if top is not None else 0.2
    bottom = bottom if bottom is not None else 0.8
    left = left if left is not None else 0.3
    right = right if right is not None else 0.7
    return top, left, bottom, right

def _generic_dimensions(rows=None, cols=None):
    rows = rows if rows is not None else 5
    cols = cols if cols is not None else 7
    return rows, cols

if __name__ == '__main__':
    pass
