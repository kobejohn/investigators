import unittest

from mock import patch
import numpy

from imagefinder import ImageFinder


class Test_ImageFinder(unittest.TestCase):
    # Initialization
    @patch.object(ImageFinder, '_standardize')
    def test___init___standardizes_image_and_mask(self, m_standardize):
        m_standardize.return_value = generic_image(channels=3)  # some valid
        img = generic_image()
        mask = generic_image()
        ImageFinder(img, mask=mask)
        # self.assertTrue(m_standardize.called)
        m_standardize.assert_called_first_with(img)
        m_standardize.assert_called_second_with(mask)

    def test___init___stores_original_size_if_no_sizes_provided(self):
        h, w = size_key = 10, 10
        img = generic_image(width=w, height=h)
        imgf = ImageFinder(img)
        self.assertTrue(size_key in imgf._templates)

    @patch.object(ImageFinder, '_mask')
    def test___init___applies_an_optional_mask_to_image(self, m_mask):
        m_mask.return_value = generic_image(channels=3)  # some valid
        img = generic_image()
        mask = generic_image()
        ImageFinder(img, mask=mask)
        # self.assertTrue(m_standardize.called)
        m_mask.assert_called()

    # Attributes and Sizing
    def test_internal_template_dict_keys_are_int_tuples_of_height_width(self):
        imgf = generic_ImageFinder()
        for height, width in imgf._templates.keys():
            self.assertIsInstance(height, int)
            self.assertIsInstance(width, int)

    def test_internal_template_dict_has_a_bgr_template_per_size(self):
        imgf = generic_ImageFinder()
        bgr_channels = 3
        for (rows, cols), template in imgf._templates.items():
            size_spec = (rows, cols, bgr_channels)
            self.assertSequenceEqual(template.shape, size_spec)

    # Locating templates in an image:
    @patch.object(ImageFinder, '_standardize')
    def test_locate_in_standardizes_the_scene_image(self, m_standardize):
        m_standardize.return_value = generic_image(channels=3)  # some valid
        # setup the image finder
        template = generic_image(width=2, height=2)
        imgf = ImageFinder(template, sizes=None)
        # confirm standardize is called when analyzing a scene
        scene = generic_image(width=10, height=10)
        imgf._standardize(scene)
        m_standardize.assert_called_with(scene)

    def test_locate_returns_None_if_templates_not_found(self):
        # setup a template and scene that should be guaranteed not to be matched
        black_template = generic_image(width=2, height=2)
        black_template.fill(0)
        white_scene = generic_image(width=10, height=10)
        white_scene.fill(255)
        imgf = ImageFinder(black_template)
        p = imgf.locate_in(white_scene)
        self.assertIsNone(p)

    def test_locate_returns_result_at_end_when_immediate_not_passed(self):
        loc_spec = loc_top, loc_left = (20, 30)
        size_spec = size_h, size_w = (100, 40)
        black_template = generic_image(width=size_w, height=size_h)
        black_template.fill(0)
        white_scene = generic_image(width=size_w * 3, height=size_h * 3)
        white_scene.fill(255)
        # set a black square to match the template
        white_scene[loc_top:loc_top + size_h,
                    loc_left: loc_left + size_w] = 0
        # setup an impossible "acceptable" threshold so it waits until end
        imgf = ImageFinder(black_template,
                           acceptable_threshold=0.1, immediate_threshold=-10)
        loc_size = imgf.locate_in(white_scene)
        self.assertIsNotNone(loc_size, 'Unexpectedly failed to return'
                                       ' a result.')
        loc, size = loc_size
        self.assertEqual((loc, size), (loc_spec, size_spec))

    def test_locate_returns_result_immediately_when_immediate_passes(self):
        loc_spec = loc_top, loc_left = (20, 30)
        size_spec = size_h, size_w = (100, 40)
        black_template = generic_image(width=size_w, height=size_h)
        black_template.fill(0)
        white_scene = generic_image(width=size_w * 3, height=size_h * 3)
        white_scene.fill(255)
        # set a black square to match the template
        white_scene[loc_top:loc_top + size_h,
                    loc_left: loc_left + size_w] = 0
        # setup an easy immediate_threshold
        imgf = ImageFinder(black_template,
                           acceptable_threshold=0.1, immediate_threshold=0.8)
        loc_size = imgf.locate_in(white_scene)
        self.assertIsNotNone(loc_size, 'Unexpectedly failed to return'
                                       ' a result.')
        loc, size = loc_size
        self.assertEqual((loc, size), (loc_spec, size_spec))

    # Internal specifications
    def test__standardize_raises_TypeError_unless_bgr_bgra_or_gray(self):
        bgr = generic_image(channels=3)
        bgra = generic_image(channels=4)
        gray = generic_image(channels=None)
        unhandled_channel_count = generic_image(channels=2)
        just_a_string = 'whut'
        # confirm the bad one fails
        imgf = generic_ImageFinder()
        self.assertRaises(TypeError,
                          imgf._standardize, unhandled_channel_count)
        self.assertRaises(TypeError,
                          imgf._standardize, just_a_string)
        # confirm the valid ones don't fail
        for ok_img in (bgr, bgra, gray):
            imgf._standardize(ok_img)

    def test__mask_puts_noise_on_masked_locations(self):
        size = h, w = 10, 10
        # make a special white image with values over random range
        high_value = 1000
        white = numpy.ndarray((h, w), dtype=numpy.uint16)
        white.fill(high_value)
        # make a mask with zeros at specified positions
        mask = numpy.ndarray((h, w), dtype=numpy.uint8)
        mask.fill(255)
        masked_positions = [(0, 0), (5, 5)]
        for p in masked_positions:
            mask[p] = 0
        # make the image finder and confirm the internal image was masked
        imgf = ImageFinder(white, mask=mask, sizes=None)
        for p in masked_positions:
            masked_pixel = imgf._templates[size][p]
            self.assertFalse(numpy.all(masked_pixel == high_value))


def generic_ImageFinder(width=None, height=None, channels=None, sizes=None):
    img = generic_image(width=width, height=height, channels=channels)
    sizes = sizes or ((5, 5), (100, 100))
    return ImageFinder(img, sizes=sizes)


def generic_image(width=None, height=None, channels=None):
    width = width or 4
    height = height or 3
    if channels:
        shape = (height, width, channels)
    else:
        shape = (height, width)
    return numpy.zeros(shape, dtype=numpy.uint8)
