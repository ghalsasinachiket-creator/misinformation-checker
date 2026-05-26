import unittest

from misinformation_checker import CLIPMisinformationDetector


class StubDetector(CLIPMisinformationDetector):
    def __init__(self, similarity, heatmap):
        super().__init__(threshold=0.35)
        self._similarity = similarity
        self._heatmap = heatmap

    def compute_similarity(self, image, caption):
        return self._similarity

    def generate_gradcam(self, image, caption):
        return self._heatmap


class DetectorTests(unittest.TestCase):
    def test_flags_misleading_pair_when_similarity_is_low(self):
        detector = StubDetector(similarity=0.2, heatmap=[[0.1, 0.9]])

        result = detector.predict(image="image", caption="caption")

        self.assertTrue(result.is_misleading)
        self.assertAlmostEqual(result.misinformation_score, 0.8)
        self.assertEqual(result.gradcam_heatmap, [[0.1, 0.9]])

    def test_does_not_flag_pair_when_similarity_is_high(self):
        detector = StubDetector(similarity=0.92, heatmap=[[0.2, 0.8]])

        result = detector.predict(image="image", caption="caption")

        self.assertFalse(result.is_misleading)
        self.assertAlmostEqual(result.misinformation_score, 0.08)

    def test_can_disable_explainability(self):
        detector = StubDetector(similarity=0.4, heatmap=[[0.5]])

        result = detector.predict(image="image", caption="caption", with_explainability=False)

        self.assertIsNone(result.gradcam_heatmap)


if __name__ == "__main__":
    unittest.main()
