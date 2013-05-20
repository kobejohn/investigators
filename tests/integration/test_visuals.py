import unittest

from mock import patch
import numpy
# silly workaround to avoid lighting up PyCharm with false errors
patch.object = patch.object

from investigators import visuals
from investigators.visuals import cv2
from investigators.visuals import ProportionalRegion, TemplateFinder, Grid
from investigators.visuals import ImageIdentifier


class Test_ImageIdentifier(unittest.TestCase):
    def test___init___standardizes_all_templates(self):
        # produce some templates
        image = _generic_image()
        templates = {'some_template': image}
        with patch.object(visuals, '_standardize_image') as m_stdize:
            m_stdize.return_value = _generic_image(channels=3)
            ImageIdentifier(templates)
        # bare mock assertions are dangerous. without confirmation, could just
        # be mispelled
        self.assertIsNone(m_stdize.assert_called_with(image))

    def test_identify_standardizes_image(self):
        # setup the image identifier
        ii = self._generic_ImageIdentifier()
        # confirm standardize is called when identifying an image
        image = _generic_image()
        with patch.object(visuals, '_standardize_image') as m_stdize:
            m_stdize.return_value = _generic_image(channels=3)
            ii.identify(image)
        # bare mock assertions are dangerous. without confirmation, could just
        # be mispelled
        self.assertIsNone(m_stdize.assert_called_with(image))

    def test_identify_equalizes_template_and_image_sizes(self):
        template = _generic_image()
        templates = {'some_template': template}
        ii = self._generic_ImageIdentifier(templates=templates)
        image = _generic_image()
        with patch.object(ii, '_equalize') as m_equalize:
            m_equalize.return_value = (_generic_image(channels=3),) * 2
            ii.identify(image)
        # bare mock assertions are dangerous. without confirmation, could just
        # be mispelled
        self.assertIs(m_equalize.called, True)

    def test_identify_returns_None_if_no_qualifying_match_found(self):
        # create a template and image that are bad matches
        template = _generic_image()
        templates = {'some_template': template}
        image = _generic_image()
        # create an identifier with both impossible thresholds
        impossible = -1
        ii = self._generic_ImageIdentifier(templates,
                                           acceptable_threshold=impossible,
                                           immediate_threshold=impossible)
        # try to identify the white image and confirm that it failed
        best_match = ii.identify(image)
        self.assertIsNone(best_match)

    def test_identify_returns_name_quickly_if_over_immediate_threshold(self):
        # create images to match
        template = _generic_image()
        template_name = 'some_template'
        templates = {template_name: template}
        image = _generic_image()
        # set the immediate threshold to easy, and acceptable to impossible
        always = 2
        impossible = -1
        ii = self._generic_ImageIdentifier(templates,
                                           acceptable_threshold=impossible,
                                           immediate_threshold=always)
        # try to identify the white image and confirm that it failed
        best_match = ii.identify(image)
        self.assertEqual(best_match, template_name)

    def test_identify_returns_name_at_end_if_over_acceptable_threshold(self):
        # create images to match
        template = _generic_image()
        template_name = 'some_template'
        templates = {template_name: template}
        image = _generic_image()
        # set the immediate threshold to easy, and acceptable to impossible
        always = 2
        impossible = -1
        ii = self._generic_ImageIdentifier(templates,
                                           acceptable_threshold=always,
                                           immediate_threshold=impossible)
        # try to identify the white image and confirm that it failed
        best_match = ii.identify(image)
        self.assertEqual(best_match, template_name)

    def test__equalize_sizes_shrinks_large_template_to_fit_in_image(self):
        template_h, template_w = 15, 100
        image_h, image_w = small_image_size = 20, 20
        shrunk_template_size_spec = 3, 20  # 15/5, 100/5
        big_template = _generic_image(height=template_h, width=template_w)
        small_image = _generic_image(height=image_h, width=image_w)
        # confirm bigger template gets shrunk; smaller image is unchanged
        ii = self._generic_ImageIdentifier()
        eq_template, eq_image = ii._equalize(big_template, small_image)
        equalized_template_size = eq_template.shape[0:2]
        equalized_image_size = eq_image.shape[0:2]
        self.assertEqual(equalized_template_size, shrunk_template_size_spec)
        self.assertEqual(equalized_image_size, small_image_size)

    def test__equalize_sizes_shrinks_large_image_so_that_template_fits(self):
        template_h, template_w = original_template_size = 5, 10
        image_h, image_w = 20, 100
        shrunk_image_size_spec = 5, 25  # 20/4, 100/4
        small_template = _generic_image(height=template_h, width=template_w)
        big_image = _generic_image(height=image_h, width=image_w)
        # confirm bigger image gets shrunk; smaller template is unchanged
        ii = self._generic_ImageIdentifier()
        eq_template, eq_image = ii._equalize(small_template, big_image)
        equalized_template_size = eq_template.shape[0:2]
        equalized_image_size = eq_image.shape[0:2]
        self.assertEqual(equalized_template_size, original_template_size)
        self.assertEqual(equalized_image_size, shrunk_image_size_spec)

    def _generic_ImageIdentifier(self, templates=None,
                                 acceptable_threshold=0.5,
                                 immediate_threshold=0.1):
        templates = templates or {str(i): _generic_image() for i in range(3)}
        return ImageIdentifier(templates,
                               acceptable_threshold=acceptable_threshold,
                               immediate_threshold=immediate_threshold)


class Test_Grid(unittest.TestCase):
    # Initialization
    def test___init___sets_grid_dimensions(self):
        some_dimensions = _generic_dimensions()
        g = self._generic_grid(dimensions=some_dimensions)
        self.assertEqual(g.dimensions, some_dimensions)

    def test___init___sets_cell_padding(self):
        padding = self._generic_padding()
        g = self._generic_grid(cell_padding=padding)
        self.assertEqual(g.cell_padding, padding)

    # Configuration
    def test_setting_dimensions_validates_them(self):
        some_dimensions = (5, 7)
        other_dimensions = (11, 13)
        g = self._generic_grid(dimensions=some_dimensions)
        with patch.object(visuals, '_validate_dimensions') as m_validate:
            g.dimensions = other_dimensions
        # bare mock assertions are dangerous. without confirmation, could just
        # be mispelled
        self.assertIsNone(m_validate.assert_called_with(other_dimensions))

    def test_seting_cell_padding_validates_padding_converted_to_rectangle(self):
        cell_padding_1 = (0, .1, .2, .3)
        cell_padding_2 = p_top, p_left, p_bottom, p_right = (.1, .2, 0, .1)
        # this is the specification of how padding is converted to a rectangle
        padded_cell_2 = visuals.Rectangle(p_top, p_left,
                                          1 - p_bottom, 1 - p_right)
        g = self._generic_grid(cell_padding=cell_padding_1)
        # Set *new* padding and make sure the validation is called
        with patch.object(visuals, '_validate_proportions') as m_validate:
            g.cell_padding = cell_padding_2
        # bare mock assertions are dangerous. without confirmation, could just
        # be mispelled
        self.assertIsNone(m_validate.assert_called_with(padded_cell_2))

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

    def _generic_padding(self, top=0.1, left=0.2, bottom=0.3, right=0.4):
        return top, left, bottom, right


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
        # bare mock assertions are dangerous. without confirmation, could just
        # be mispelled
        self.assertIsNone(m_validate.assert_called_with(other_proportions))

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
    def test___init___standardizes_template(self):
        img = _generic_image()
        mask = _generic_image()
        with patch.object(visuals, '_standardize_image') as m_stdize:
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
        with patch.object(visuals, '_standardize_image') as m_stdize:
            m_stdize.return_value = _generic_image(channels=3)
            imgf.locate_in(scene)
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
    def test__standardize_mask_raise_TypeError_unless_bgr_bgra_or_gry(self):
        bgr = _generic_image(channels=3)
        bgra = _generic_image(channels=4)
        gray = _generic_image(channels=None)
        unhandled_channel_count = _generic_image(channels=2)
        just_a_string = 'whut'
        # confirm the bad one fails
        imgf = self._generic_ImageFinder()
        self.assertRaises(TypeError, imgf._standardize_mask,
                          unhandled_channel_count)
        self.assertRaises(TypeError, imgf._standardize_mask,
                          just_a_string)
        # confirm the valid ones don't fail
        for ok_img in (bgr, bgra, gray):
            imgf._standardize_mask(ok_img)

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

    def test__standardize_image_raise_TypeError_unless_bgr_bgra_or_gry(self):
        bgr = _generic_image(channels=3)
        bgra = _generic_image(channels=4)
        gray = _generic_image(channels=None)
        unhandled_channel_count = _generic_image(channels=2)
        just_a_string = 'whut'
        # confirm the bad one fails
        self.assertRaises(TypeError, visuals._standardize_image,
                          unhandled_channel_count)
        self.assertRaises(TypeError, visuals._standardize_image,
                          just_a_string)
        # confirm the valid ones don't fail
        for ok_img in (bgr, bgra, gray):
            visuals._standardize_image(ok_img)


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
