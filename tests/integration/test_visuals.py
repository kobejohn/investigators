import unittest
from os import path

from mock import patch
import numpy
# silly workaround to avoid lighting up PyCharm with false errors
patch.object = patch.object

from investigators import visuals  # for testing module functions
from investigators.visuals import cv2
from investigators.visuals import ProportionalRegion, TemplateFinder, Grid
from investigators.visuals import ImageIdentifier, TankLevel

this_path = path.abspath(path.split(__file__)[0])


class Test_TankLevel(unittest.TestCase):
    # Primary behavior specification
    def test_how_full_returns_approximatelly_correct_fill_level(self):
        tank_image = cv2.imread(path.join(this_path, 'health bar 77%.png'))
        self.assertIsNotNone(tank_image)  # to avoid confusing errors
        fill = (5, 5, 200)
        empty = (40, 40, 50)
        ignore = (20, 20, 20)
        tl = self._generic_TankLevel(fill=fill, empty=empty, ignore=ignore)
        fill_level = tl.how_full(tank_image)
        fill_level_spec = 0.77
        tolerance = 0.2  # allow +/- 20%
        self.assertAlmostEqual(fill_level, fill_level_spec, delta=tolerance)

    def test_how_full_returns_full_if_fill_is_majority_and_no_border(self):
        fill = 255
        empty = 0
        tank_image = _generic_image()
        tank_image.fill(fill)
        tl = self._generic_TankLevel(fill=fill, empty=empty)
        fill_level = tl.how_full(tank_image)
        fill_level_spec = 1.0
        tolerance = 0.1  # allow +/- 10%
        self.assertAlmostEqual(fill_level, fill_level_spec, delta=tolerance)

    def test_how_full_returns_empty_if_fill_is_majority_and_no_border(self):
        fill = 255
        empty = 0
        tank_image = _generic_image()
        tank_image.fill(empty)
        tl = self._generic_TankLevel(fill=fill, empty=empty)
        fill_level = tl.how_full(tank_image)
        fill_level_spec = 0.0
        tolerance = 0.1  # allow +/- 10%
        self.assertAlmostEqual(fill_level, fill_level_spec, delta=tolerance)

    def test_how_full_returns_None_if_crop_has_an_error(self):
        tl = self._generic_TankLevel(ignore=(0, 0, 0))
        with patch.object(tl, '_crop') as m_crop:
            m_crop.return_value = None
            fill_level = tl.how_full(_generic_image())
        self.assertIsNone(fill_level)

    # Configuration / Internal specification
    def test_how_full_standardizes_tank_image(self):
        # setup the image identifier
        tl = self._generic_TankLevel()
        # confirm standardize is called when measuring the tank
        image = _generic_image()
        with patch.object(visuals, '_standardize_image') as m_stdize:
            m_stdize.return_value = _generic_image(channels=3)
            tl.how_full(image)
        self.assertIsNone(m_stdize.assert_called_with(image))

    def test_setting_region_colors_validates_them(self):
        colors = fill, empty, ignore = (0, 0, 0), (10, 10, 10), (20, 20, 20)
        tl = self._generic_TankLevel(fill=fill, empty=empty, ignore=ignore)
        with patch.object(tl, '_validate_colors') as m_validate:
            tl.colors = fill, empty, ignore
        self.assertIsNone(m_validate.assert_called_with(colors))

    def test__validate_colors_raises_TypeError_unless_3_item_tuple(self):
        only_2_colors = (1, 10)
        tl = self._generic_TankLevel()
        self.assertRaises(TypeError, setattr, *(tl, 'colors', only_2_colors))

    def test__validate_colors_raises_ValueError_if_any_2_colors_are_equal(self):
        same_colors = ((0, 0, 0), (0, 0, 0), (10, 10, 10))
        tl = self._generic_TankLevel()
        self.assertRaises(ValueError, setattr, *(tl, 'colors', same_colors))

    def test__validate_colors_raises_ValueError_if_fill_or_empty_is_None(self):
        ignore_is_None = ((0, 0, 0), (1, 1, 1), None)
        fill_is_None = (None, (1, 1, 1), (2, 2, 2))
        empty_is_None = ((0, 0, 0), None, (2, 2, 2))
        tl = self._generic_TankLevel()
        self.assertRaises(ValueError, setattr, *(tl, 'colors', fill_is_None))
        self.assertRaises(ValueError, setattr, *(tl, 'colors', empty_is_None))
        try:
            tl.colors = ignore_is_None
        except Exception as e:
            self.fail('Unexpectedly failed to allow ignore to be None:\n{}'
                      ''.format(e))

    def _generic_TankLevel(self, fill=None, empty=None, ignore=None):
        fill = fill if fill is not None else (0, 0, 255)
        empty = empty if empty is not None else (0, 0, 50)
        ignore = ignore
        return TankLevel(fill, empty, ignore)


class Test_screen_shot(unittest.TestCase):
    # Primary behavior specification
    def test_screen_shot_gets_an_image_with_same_resolution_as_screen(self):
        # thanks to jcao219 for the python-only way to get screen resolution
        # http://stackoverflow.com/a/3129524/377366
        import ctypes
        user32 = ctypes.windll.user32
        ss_h_w_spec = user32.GetSystemMetrics(1), user32.GetSystemMetrics(0)
        # get the actual screen shot and confirm the size
        screen_shot = visuals.screen_shot()
        ss_h_w = screen_shot.shape[0:2]
        self.assertEqual(ss_h_w, ss_h_w_spec)

    # Configuration / Internal specification
    def test_screen_shot_standardizes_the_screen_image(self):
        import ImageGrab
        with patch.object(ImageGrab, 'grab') as m_grab:
            m_grab.return_value = _generic_image()
            with patch.object(visuals, '_standardize_image') as m_stdize:
                m_stdize.return_value = _generic_image(channels=3)
                visuals.screen_shot()
        self.assertIsNone(m_stdize.assert_called_with(m_grab.return_value))


class Test_ImageIdentifier(unittest.TestCase):
    # Primary behavior specification
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

    # Configuration / Internal specification
    def test___init___standardizes_all_templates(self):
        # produce some templates
        image = _generic_image()
        templates = {'some_template': image}
        with patch.object(visuals, '_standardize_image') as m_stdize:
            m_stdize.return_value = _generic_image(channels=3)
            ImageIdentifier(templates)
        self.assertIsNone(m_stdize.assert_called_with(image))

    def test_identify_standardizes_image(self):
        # setup the image identifier
        ii = self._generic_ImageIdentifier()
        # confirm standardize is called when identifying an image
        image = _generic_image()
        with patch.object(visuals, '_standardize_image') as m_stdize:
            m_stdize.return_value = _generic_image(channels=3)
            ii.identify(image)
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

    def test__equalize_shrinks_large_template_to_fit_in_image(self):
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

    def test__equalize_shrinks_large_image_so_that_template_fits(self):
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
    # Primary behavior specification
    def test_gridify_generates_correct_sequence_of_borders(self):
        # get the test image which is constructed as follows:
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

    # Configuration / Internal specification
    def test___init___sets_grid_dimensions(self):
        some_dimensions = (5, 7)
        g = self._generic_grid(dimensions=some_dimensions)
        self.assertEqual(g.dimensions, some_dimensions)

    def test___init___sets_cell_padding(self):
        padding = (0.1, 0.2, 0.3, 0.4)
        g = self._generic_grid(cell_padding=padding)
        self.assertEqual(g.cell_padding, padding)

    def test_setting_dimensions_validates_them(self):
        some_dimensions = (5, 7)
        other_dimensions = (11, 13)
        g = self._generic_grid(dimensions=some_dimensions)
        with patch.object(visuals, '_validate_dimensions') as m_validate:
            g.dimensions = other_dimensions
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
        self.assertIsNone(m_validate.assert_called_with(padded_cell_2))

    def _generic_grid(self, dimensions=None, cell_padding=None):
        dimensions = dimensions or (5, 7)
        cell_padding = cell_padding or (0.1, 0.3, 0.5, 0.7)
        return Grid(dimensions, cell_padding)


class Test_ProportionalRegion(unittest.TestCase):
    # Primary behavior specification
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

    # Configuration / Internal specification
    def test___init___sets_proportions(self):
        some_proportions = _generic_proportions()
        pr = ProportionalRegion(some_proportions)
        self.assertEqual(pr.proportions, some_proportions)

    def test_seting_proportions_validates_them(self):
        some_proportions = (0, .1, .2, .3)
        other_proportions = (.4, .5, .6, .7)
        pr = ProportionalRegion(some_proportions)
        with patch.object(visuals, '_validate_proportions') as m_validate:
            pr.proportions = other_proportions
        self.assertIsNone(m_validate.assert_called_with(other_proportions))


class Test_TemplateFinder(unittest.TestCase):
    # Primary behavior specification
    def test_locate_returns_None_if_no_internal_templates_found_in_scene(self):
        # setup a template and scene that should be guaranteed not to be matched
        black_template = _generic_image(height=5, width=10)
        black_template.fill(0)
        white_scene = _generic_image(height=20, width=20)
        white_scene.fill(255)
        tf = TemplateFinder(black_template)
        p = tf.locate_in(white_scene)
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
        tf = TemplateFinder(black_template,
                            acceptable_threshold=always,
                            immediate_threshold=impossible)
        borders = tf.locate_in(white_scene)
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
        tf = TemplateFinder(black_template, acceptable_threshold=impossible,
                            immediate_threshold=always)
        # confirm the result
        borders = tf.locate_in(white_scene)
        self.assertEqual(borders,
                         (top_spec, left_spec, bottom_spec, right_spec))

    # Configuration / Internal specification
    def test___init___changes_parts_with_templ_mask_sizes_scale(self):
        template = _generic_image()
        mask = _generic_image()
        sizes = ((10, 20),)
        scale = .9876
        kwargs = {'template': template,
                  'mask': mask,
                  'sizes': sizes,
                  'scale_for_speed': scale}
        with patch.object(TemplateFinder, 'change_parts') as m_change_parts:
            self._generic_TemplateFinder(**kwargs)
        self.assertIsNone(m_change_parts.assert_called_with(**kwargs))

    def test_change_parts_standardizes_template_and_mask(self):
        tf = self._generic_TemplateFinder()
        with patch.object(visuals, '_standardize_image') as m_standardize:
            m_standardize.return_value = _generic_image()
            template = _generic_image()
            mask = _generic_image()
            tf.change_parts(template=template, mask=mask)
        self.assertEqual(m_standardize.call_count, 2)

    def test_change_parts_validates_sizes(self):
        tf = self._generic_TemplateFinder()
        with patch.object(tf, '_validate_sizes') as m_validate:
            sizes = ((20, 10), (30, 20))
            tf.change_parts(sizes=sizes)
        self.assertIsNone(m_validate.assert_any_call(sizes))

    def test_change_parts_validates_scale(self):
        tf = self._generic_TemplateFinder()
        with patch.object(tf, '_validate_scale_for_speed') as m_validate:
            scale = 0.54321
            tf.change_parts(scale_for_speed=scale)
        self.assertIsNone(m_validate.assert_any_call(scale))

    def test_change_parts_builds_new_templates(self):
        tf = self._generic_TemplateFinder()
        with patch.object(tf, '_build_templates') as m_build:
            tf.change_parts(scale_for_speed=0.54321)
        self.assertIsNone(m_build.assert_called_with())

    def test_change_parts_sets_template_mask_sizes_and_scale(self):
        tf = self._generic_TemplateFinder()
        original_template = tf.template
        original_mask = tf.mask
        original_sizes = tf.sizes
        original_scale = tf.scale_for_speed
        tf.change_parts(template=_generic_image(), mask=_generic_image(),
                        sizes=((20, 10),), scale_for_speed=0.1234)
        # confirm that all the parts have changed
        self.assertIsNot(tf.template, original_template)
        self.assertIsNot(tf.mask, original_mask)
        self.assertIsNot(tf.sizes, original_sizes)
        self.assertIsNot(tf.scale_for_speed, original_scale)

    def test_template_and_mask_and_sizes_and_scale_are_read_only(self):
        tf = self._generic_TemplateFinder()
        read_only_names = ('template', 'mask', 'sizes', 'scale_for_speed')
        for name in read_only_names:
            self.assertRaises(AttributeError, setattr, *(tf, name, 1))

    def test__validate_scale_raises_ValueError_if_lte_0_or_gt_1(self):
        tf = self._generic_TemplateFinder()
        zero = 0
        negative = -0.5
        over_one = 1.1
        ok = 0.5
        try:
            tf._validate_scale_for_speed(ok)
        except Exception as e:
            self.fail('Unexpectedly failed with a valid scale {}:\n{}'
                      ''.format(ok, e))
        for bad_scale in (zero, negative, over_one):
            self.assertRaises(ValueError,
                              tf._validate_scale_for_speed, bad_scale)

    def test__build_templates_applies_the_mask_to_the_base_template(self):
        tf = self._generic_TemplateFinder()
        template = tf.template
        mask = tf.mask
        with patch.object(tf, '_apply_mask') as m_apply_mask:
            m_apply_mask.return_value = _generic_image()
            tf._build_templates()
        self.assertIsNone(m_apply_mask.assert_called_with(*(template, mask)))

    def test__mask_puts_noise_on_masked_locations(self):
        h, w = 20, 10
        # make a special white image with values over random range
        high_value = 1000
        white = numpy.ndarray((h, w, 3), dtype=numpy.uint16)
        white.fill(high_value)
        # make a mask with zeros at specified positions
        mask = numpy.ndarray((h, w, 3), dtype=numpy.uint8)
        mask.fill(255)
        masked_positions = [(0, 0), (5, 5)]
        for p in masked_positions:
            mask[p] = 0
        # use a generic image finder to test _mask
        tf = self._generic_TemplateFinder()
        masked_white = tf._apply_mask(white, mask)
        for p in masked_positions:
            masked_pixel = masked_white[p]
            # confirm all noise pixels have been changed
            self.assertTrue(numpy.all(masked_pixel != high_value))

    def test__build_templates_scales_the_masked_template(self):
        new_h, new_w = 3, 4
        tf = self._generic_TemplateFinder(sizes=((new_h, new_w),))
        with patch.object(tf, '_apply_mask') as m_apply_mask:
            m_apply_mask.return_value = masked_image = _generic_image()
            with patch.object(cv2, 'resize') as m_resize:
                tf._build_templates()
        args = (masked_image, (new_w, new_h))
        kwargs = {'interpolation': cv2.INTER_AREA}
        self.assertIsNone(m_resize.assert_called_with(*args, **kwargs))

    def test__validate_sizes_allows_None_and_2_real_tuple_sequences_only(self):
        none = None
        sequence_with_2_tuples = ((1, 2), (3, 4))
        sequence_with_2_tuple_floats = ((1.0, 2.0), (3.0, 4.0))
        sequence_with_3_tuple = ((1, 3, 3),)
        no_sequence_2_tuple = (1, 2)
        ok = (none, sequence_with_2_tuples, sequence_with_2_tuple_floats)
        bad = (sequence_with_3_tuple, no_sequence_2_tuple)
        tf = self._generic_TemplateFinder()
        for ok_sizes in ok:
            try:
                tf._validate_sizes(ok_sizes)
            except Exception as e:
                self.fail('Unexpectedly failed to validate valid sizes:{}\n{}'
                          ''.format(ok_sizes, e))
        for bad_sizes in bad:
            self.assertRaises(Exception, tf._validate_sizes, bad_sizes)

    def test__build_templates_stores_scaled_templates_under_original_size(self):
        sizes_spec = ((10, 20), (30, 40))
        scale = 0.5
        tf = self._generic_TemplateFinder(sizes=sizes_spec,
                                          scale_for_speed=scale)
        # confirm that the stored template keys are not scaled
        self.assertItemsEqual(tf._templates.keys(), sizes_spec)

    def test_locate_in_standardizes_the_scene_image(self):
        tf = self._generic_TemplateFinder()
        # confirm standardize is called when analyzing a scene
        scene = _generic_image()
        with patch.object(visuals, '_standardize_image') as m_stdize:
            m_stdize.return_value = _generic_image(channels=3)
            tf.locate_in(scene)
        self.assertIsNone(m_stdize.assert_called_with(scene))

    def test_locate_in_scales_the_scene_by_scale_for_speed(self):
        scene_base_h, scene_base_w = 300, 400
        scale = 0.5
        tf = self._generic_TemplateFinder(scale_for_speed=scale)
        with patch.object(visuals, '_standardize_image') as m_stdize:
            # make a standardized scene of the base size
            # that sould be resized by the scale to the specified size
            scene_std = _generic_image(width=scene_base_w, height=scene_base_h)
            m_stdize.return_value = scene_std
            with patch.object(cv2, 'resize') as m_resize:
                m_resize.return_value = _generic_image()
                tf.locate_in(_generic_image())
        final_h, final_w = (150, 200)
        args = (scene_std, (final_w, final_h))
        kwargs = {'interpolation': cv2.INTER_AREA}
        self.assertIsNone(m_resize.assert_any_call(*args, **kwargs))

    def test_locate_in_ignores_templates_too_big_for_the_scene(self):
        # setup an image finder with only a large size for template
        large_h, large_w = 100, 200
        tf = self._generic_TemplateFinder(sizes=((large_h, large_w),))
        # setup a scene smaller than the large size template
        small_h, small_w = large_h - 20, large_w - 20
        small_scene = _generic_image(height=small_h, width=small_w)
        # confirm that matchTemplate is not called
        with patch.object(cv2, 'matchTemplate') as m_match:
            m_match.return_value = _generic_image(channels=None)
            tf.locate_in(small_scene)
        self.assertFalse(m_match.called)

    def _generic_TemplateFinder(self, template=None, mask=None,
                                sizes=None, scale_for_speed=1,
                                acceptable_threshold=0.5,
                                immediate_threshold=0.1):
        template = template if template is not None else _generic_image()
        return TemplateFinder(template, mask=mask,
                              sizes=sizes, scale_for_speed=scale_for_speed,
                              acceptable_threshold=acceptable_threshold,
                              immediate_threshold=immediate_threshold)


class Test_Module_Helpers(unittest.TestCase):
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

    def test__standardize_image_raise_TypeErr_unless_bgr_bgra_gry_or_PIL(self):
        import Image
        pil = Image.new('RGB', (4,3))
        bgr = _generic_image(channels=3)
        bgra = _generic_image(channels=4)
        gray = _generic_image(channels=None)
        unhandled_channel_count = _generic_image(channels=2)
        just_a_string = 'whut'
        # confirm the bad ones fail
        for bad_img in (unhandled_channel_count, just_a_string):
            self.assertRaises(TypeError, visuals._standardize_image, bad_img)
        # confirm the valid ones don't fail
        for ok_img in (pil, bgr, bgra, gray):
            visuals._standardize_image(ok_img)


# Helper factories
def _generic_image(height=None, width=None, channels=None):
    height = height if not height is None else 30
    width = width if not width is None else 40
    if channels is 0:
        shape = (height, width)
    else:
        shape = (height, width, channels or 3)
    return numpy.zeros(shape, dtype=numpy.uint8)


def _generic_proportions(top=None, left=None, bottom=None, right=None):
    top = top if top is not None else 0.2
    bottom = bottom if bottom is not None else 0.8
    left = left if left is not None else 0.3
    right = right if right is not None else 0.7
    return top, left, bottom, right


if __name__ == '__main__':
    pass
