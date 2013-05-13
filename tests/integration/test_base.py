import unittest

import numpy

from imagefinder import ImageFinder


class Test_ImageFinder(unittest.TestCase):
    # Initialization
    def test___init___raises_TypeError_unless_numpy_bgr_bgra_or_gray(self):
        bgr = generic_image(channels=3)
        bgra = generic_image(channels=4)
        gray = generic_image(channels=1)
        bad = generic_image(channels=2)
        totally_wrong = 'whut'
        # confirm the bad one fails
        self.assertRaises(TypeError, ImageFinder, bad)
        self.assertRaises(TypeError, ImageFinder, totally_wrong)
        # confirm the good ones don't fail
        for ok_img in (bgr, bgra, gray):
            try:
                ImageFinder(ok_img)
            except Exception as e:
                self.fail('Unexpectedly failed to initialize with'
                          ' a valid image:\n{}.'.format(e))

    def test___init___stores_base_image_as_template_if_no_sizes_provided(self):
        h, w = size_key = 10, 10
        img = generic_image(width=w, height=h)
        imgf = ImageFinder(img)
        self.assertIs(imgf._templates[size_key], imgf._base)

    def test___init___stores_image_internally_as_bgr(self):
        bgra_channels = 4
        imgf = generic_ImageFinder(channels=bgra_channels)
        internal_image = imgf._base
        bgr_channels = 3
        self.assertEqual(internal_image.shape[2], bgr_channels)

    def test___init___applies_an_optional_mask_to_internal_image(self):
        h, w = 10, 10
        # make a special white image with values over random range
        high_value = 1000
        white_img = numpy.ndarray((h, w), dtype=numpy.uint16)
        white_img.fill(high_value)
        # make a mask with zeros at specified positions
        mask_img = numpy.ndarray((h, w), dtype=numpy.uint8)
        mask_img.fill(255)
        masked_positions = [(0, 0), (5, 5)]
        for p in masked_positions:
            mask_img[p] = 0  # 0 ==> mask
        # make the image finder and confirm the internal image was masked
        imgf = ImageFinder(white_img, mask=mask_img)
        for p in masked_positions:
            masked_pixel = imgf._base[p]
            self.assertFalse(numpy.all(masked_pixel == high_value))

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

    # Searching for templates:
    def test_locate_returns_l_after_all_comparisons_if_over_min_threshold(self):
        raise NotImplementedError

    def test_locate_returns_l_immediately_if_over_immediate_threshold(self):
        raise NotImplementedError


def generic_ImageFinder(width=None, height=None, channels=None, sizes=None):
    img = generic_image(width=width, height=height, channels=channels)
    sizes = sizes or ((5, 5), (100, 100))
    return ImageFinder(img, sizes)


def generic_image(width=None, height=None, channels=None):
    width = width or 4
    height = height or 3
    channels = channels or 3
    return numpy.zeros((height, width, channels), dtype=numpy.uint8)
