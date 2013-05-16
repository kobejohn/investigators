import unittest

from mock import patch
import numpy

from investigators import visuals
from investigators.visuals import cv2
from investigators.visuals import ProportionalRegion, TemplateFinder


class Test_ProportionalRegion(unittest.TestCase):
    # Initialization
    def test___init___sets_proportions(self):
        some_proportions = _generic_proportions()
        with patch.object(ProportionalRegion, '_set_proportions') as m_set_p:
            ProportionalRegion(some_proportions)
        self.assertTrue(m_set_p.called_with(*some_proportions))

    # Configuration
    def test_seting_proportions_validates_them(self):
        some_proportions = _generic_proportions(0, .1, .2, .3)
        other_proportions = _generic_proportions(.4, .5, .6, .7)
        pr = ProportionalRegion(some_proportions)
        with patch.object(visuals, '_validate_proportions') as m_validate:
            pr.proportions = other_proportions
        self.assertTrue(m_validate.called_with(some_proportions))

    # Helper methods
    def test__validate_proportions_raises_ValueError_if_opp_borders_rvrsd(self):
        v = 0.5
        left_right_same = _generic_proportions(left=v, right=v)
        top_bottom_same = _generic_proportions(top=v, bottom=v)
        self.assertRaises(ValueError, ProportionalRegion, left_right_same)
        self.assertRaises(ValueError, ProportionalRegion, top_bottom_same)

    def test__validate_proportions_raises_ValueError_if_borders_OOB(self):
        under_bound = -0.1
        over_bound = 1.1
        top_out = _generic_proportions(top=under_bound)
        left_out = _generic_proportions(left=under_bound)
        bottom_out = _generic_proportions(bottom=over_bound)
        right_out = _generic_proportions(right=over_bound)
        for out_of_bounds in (top_out, left_out, bottom_out, right_out):
            self.assertRaises(ValueError, ProportionalRegion, out_of_bounds)

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
    @patch.object(TemplateFinder, '_standardize_image')
    def test___init___standardizes_image(self, m_standardize_img):
        m_standardize_img.return_value = _generic_image(channels=3)
        img = _generic_image()
        mask = _generic_image()
        TemplateFinder(img, mask=mask)
        self.assertTrue(m_standardize_img.called_with(img))

    @patch.object(TemplateFinder, '_standardize_mask')
    def test___init___standardizes_mask(self, m_standardize_mask):
        m_standardize_mask.return_value = _generic_image(channels=None)  # valid
        img = _generic_image()
        mask = _generic_image()
        TemplateFinder(img, mask=mask)
        self.assertTrue(m_standardize_mask.called_with(mask))

    def test___init___stores_original_size_if_no_sizes_provided(self):
        h, w = size_key = 20, 10
        img = _generic_image(height=h, width=w)
        imgf = TemplateFinder(img)
        self.assertTrue(size_key in imgf._templates)

    @patch.object(TemplateFinder, '_mask')
    def test___init___applies_an_optional_mask_to_image(self, m_mask):
        m_mask.return_value = _generic_image(channels=3)  # some valid
        img = _generic_image()
        mask = _generic_image()
        TemplateFinder(img, mask=mask)
        self.assertTrue(m_mask.called)

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
    @patch.object(TemplateFinder, '_standardize_image')
    def test_locate_in_standardizes_the_scene_image(self, m_standardize_img):
        m_standardize_img.return_value = _generic_image(channels=3)
        # setup the image finder
        template = _generic_image(height=2, width=2)
        imgf = TemplateFinder(template, sizes=None)
        # confirm standardize is called when analyzing a scene
        scene = _generic_image(height=100, width=100)
        imgf._standardize_image(scene)
        self.assertTrue(m_standardize_img.called_with(scene))

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

    # patch with some random correlation result image
    @patch.object(cv2, 'matchTemplate')
    def test_locate_in_ignores_templates_too_big_for_the_scene(self, m_match):
        m_match.return_value = _generic_image(channels=None)
        # setup an image finder with only a large size for template
        large_h, large_w = 100, 200
        imgf = self._generic_ImageFinder(sizes=((large_h, large_w),))
        # setup a scene smaller than the large size template
        small_h, small_w = large_h - 20, large_w - 20
        small_scene = _generic_image(height=small_h, width=small_w)
        # confirm that matchTemplate is not called
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


if __name__ == '__main__':
    pass
