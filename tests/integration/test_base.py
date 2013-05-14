import unittest

from mock import patch
import cv2, numpy

from imagefinder import ImageFinder


class Test_ImageFinder(unittest.TestCase):
    # Initialization
    @patch.object(ImageFinder, '_standardize_img')
    def test___init___standardizes_image(self, m_standardize_img):
        m_standardize_img.return_value = generic_image(channels=3)  # some valid
        img = generic_image()
        mask = generic_image()
        ImageFinder(img, mask=mask)
        self.assertTrue(m_standardize_img.called_with(img))

    @patch.object(ImageFinder, '_standardize_mask')
    def test___init___standardizes_mask(self, m_standardize_mask):
        m_standardize_mask.return_value = generic_image(channels=None)  # valid
        img = generic_image()
        mask = generic_image()
        ImageFinder(img, mask=mask)
        self.assertTrue(m_standardize_mask.called_with(mask))

    def test___init___stores_original_size_if_no_sizes_provided(self):
        h, w = size_key = 20, 10
        img = generic_image(width=w, height=h)
        imgf = ImageFinder(img)
        self.assertTrue(size_key in imgf._templates)

    @patch.object(ImageFinder, '_mask')
    def test___init___applies_an_optional_mask_to_image(self, m_mask):
        m_mask.return_value = generic_image(channels=3)  # some valid
        img = generic_image()
        mask = generic_image()
        ImageFinder(img, mask=mask)
        self.assertTrue(m_mask.called)

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

    # Locating templates in a scene:
    @patch.object(ImageFinder, '_standardize_img')
    def test_locate_in_standardizes_the_scene_image(self, m_standardize_img):
        m_standardize_img.return_value = generic_image(channels=3)  # some valid
        # setup the image finder
        template = generic_image(width=2, height=2)
        imgf = ImageFinder(template, sizes=None)
        # confirm standardize is called when analyzing a scene
        scene = generic_image(width=100, height=100)
        imgf._standardize_img(scene)
        self.assertTrue(m_standardize_img.called_with(scene))

    def test_locate_returns_None_if_no_internal_templates_found_in_scene(self):
        # setup a template and scene that should be guaranteed not to be matched
        black_template = generic_image(width=10, height=5)
        black_template.fill(0)
        white_scene = generic_image(width=20, height=20)
        white_scene.fill(255)
        imgf = ImageFinder(black_template)
        p = imgf.locate_in(white_scene)
        self.assertIsNone(p)

    def test_locate_returns_result_at_end_when_immediate_not_passed(self):
        loc_spec = loc_top, loc_left = (2, 3)
        size_spec = size_h, size_w = (5, 20)
        black_template = generic_image(width=size_w, height=size_h)
        black_template.fill(0)
        white_scene = generic_image(width=size_w * 3, height=size_h * 3)
        white_scene.fill(255)
        # set a black square to match the template
        white_scene[loc_top:loc_top + size_h,
                    loc_left: loc_left + size_w] = 0
        # setup an impossible immediate threshold so it waits until end
        imgf = ImageFinder(black_template,
                           acceptable_threshold=2, immediate_threshold=-10)
        loc_size = imgf.locate_in(white_scene)
        self.assertIsNotNone(loc_size, 'Unexpectedly failed to return'
                                       ' a result.')
        loc, size = loc_size
        self.assertEqual((loc, size), (loc_spec, size_spec))

    def test_locate_returns_result_immediately_when_immediate_passes(self):
        loc_spec = loc_top, loc_left = (2, 3)
        size_spec = size_h, size_w = (5, 20)
        black_template = generic_image(width=size_w, height=size_h)
        black_template.fill(0)
        white_scene = generic_image(width=size_w * 3, height=size_h * 3)
        white_scene.fill(255)
        # set a black square to match the template
        white_scene[loc_top:loc_top + size_h,
                    loc_left: loc_left + size_w] = 0
        # setup an easy immediate_threshold and impossible acceptable threshold
        imgf = ImageFinder(black_template,
                           acceptable_threshold=-10, immediate_threshold=0.8)
        loc_size = imgf.locate_in(white_scene)
        self.assertIsNotNone(loc_size, 'Unexpectedly failed to return'
                                       ' a result.')
        loc, size = loc_size
        self.assertEqual((loc, size), (loc_spec, size_spec))

    # patch with some random correlation result image
    @patch.object(cv2, 'matchTemplate')
    def test_locate_in_ignores_templates_too_big_for_the_scene(self, m_match):
        m_match.return_value = generic_image(channels=None)  # for graceful fail
        # setup an image finder with only a large size for template
        large_h, large_w = 100, 200
        imgf = generic_ImageFinder(sizes=((large_h, large_w),))
        # setup a scene smaller than the large size template
        small_h, small_w = large_h - 20, large_w - 20
        small_scene = generic_image(height=small_h, width=small_w)
        # confirm that matchTemplate is not called
        imgf.locate_in(small_scene)
        self.assertFalse(m_match.called)

    # Internal specifications
    def test__standardize_img_mask_raise_TypeError_unless_bgr_bgra_or_gry(self):
        bgr = generic_image(channels=3)
        bgra = generic_image(channels=4)
        gray = generic_image(channels=None)
        unhandled_channel_count = generic_image(channels=2)
        just_a_string = 'whut'
        # confirm the bad one fails
        imgf = generic_ImageFinder()
        for standardizer in (imgf._standardize_img, imgf._standardize_mask):
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
        imgf = generic_ImageFinder()
        masked_white = imgf._mask(white, mask)
        for p in masked_positions:
            masked_pixel = masked_white[p]
            # confirm all noise pixels have been changed
            self.assertTrue(numpy.all(masked_pixel != high_value))


def generic_ImageFinder(width=None, height=None, channels=None, sizes=None):
    img = generic_image(width=width, height=height, channels=channels)
    sizes = sizes or ((15, 20), (30, 40))
    return ImageFinder(img, sizes=sizes)


def generic_image(width=None, height=None, channels=None):
    height = height or 30
    width = width or 40
    if channels:
        shape = (height, width, channels)
    else:
        shape = (height, width)
    return numpy.zeros(shape, dtype=numpy.uint8)
