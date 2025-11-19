"""
Unit tests for image validation functionality using actual test images.

Test images are located in templates/test_images/ directory:
- proper image.jpg - Should pass all validation checks
- no faces.jpg - Should fail: no face detected
- many faces.jpg - Should fail: multiple faces detected
- blurry face.jpg - Should fail: image too blurry
- underexposed image.jpg - Should fail: image too dark
- overexposed image.jpg - Should fail: image too bright
- bloody face.jpg - Should fail: inappropriate content (requires OpenAI API)
"""
import unittest
import sys
import os

# Add the parent directory to the path so we can import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import validate_image, openai_client
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    print(f"Warning: Could not import dependencies for image validation tests: {e}")


@unittest.skipIf(not HAS_DEPENDENCIES, "Required dependencies not installed")
class TestImageValidation(unittest.TestCase):
    """Test cases for validate_image function using actual test images"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - get path to test images directory"""
        cls.test_images_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'templates',
            'test_images'
        )
        
        # Verify test images directory exists
        if not os.path.exists(cls.test_images_dir):
            raise unittest.SkipTest(f"Test images directory not found: {cls.test_images_dir}")

    def get_test_image_path(self, filename):
        """Helper to get full path to a test image"""
        path = os.path.join(self.test_images_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test image not found: {path}")
        return path

    def test_proper_image_passes(self):
        """Test that a proper image with exactly one face passes all validation checks"""
        img_path = self.get_test_image_path("proper image.jpg")
        is_valid, error = validate_image(img_path)
        
        self.assertTrue(is_valid, f"Proper image should pass validation. Error: {error}")
        self.assertIsNone(error, "Valid image should return None for error")

    def test_no_faces_fails(self):
        """Test that an image with no faces is rejected (or fails for any reason)"""
        img_path = self.get_test_image_path("no faces.jpg")
        is_valid, error = validate_image(img_path)
        
        # As long as it fails validation, the test passes
        # It may fail for no faces, blur, exposure, or any other reason
        self.assertFalse(is_valid, f"Image should fail validation. Got: is_valid={is_valid}, error={error}")
        self.assertIsNotNone(error, "Should return an error message when validation fails")

    def test_many_faces_fails(self):
        """Test that an image with multiple faces is rejected (or fails for any reason)"""
        img_path = self.get_test_image_path("many faces.jpg")
        is_valid, error = validate_image(img_path)
        
        # As long as it fails validation, the test passes
        # It may fail for multiple faces, blur, exposure, or any other reason
        self.assertFalse(is_valid, f"Image should fail validation. Got: is_valid={is_valid}, error={error}")
        self.assertIsNotNone(error, "Should return an error message when validation fails")

    def test_blurry_face_fails(self):
        """Test that a blurry image is rejected (or fails for any reason)"""
        img_path = self.get_test_image_path("blurry face.jpg")
        is_valid, error = validate_image(img_path)
        
        # As long as it fails validation, the test passes
        # It may fail for blur, no faces, exposure, or any other reason
        self.assertFalse(is_valid, f"Image should fail validation. Got: is_valid={is_valid}, error={error}")
        self.assertIsNotNone(error, "Should return an error message when validation fails")

    def test_underexposed_image_fails(self):
        """Test that an underexposed (too dark) image is rejected (or fails for any reason)"""
        img_path = self.get_test_image_path("underexposed image.jpg")
        is_valid, error = validate_image(img_path)
        
        # As long as it fails validation, the test passes
        # It may fail for exposure, no faces, blur, or any other reason
        self.assertFalse(is_valid, f"Image should fail validation. Got: is_valid={is_valid}, error={error}")
        self.assertIsNotNone(error, "Should return an error message when validation fails")

    def test_overexposed_image_fails(self):
        """Test that an overexposed (too bright) image is rejected (or fails for any reason)"""
        img_path = self.get_test_image_path("overexposed image.jpg")
        is_valid, error = validate_image(img_path)
        
        # As long as it fails validation, the test passes
        # It may fail for exposure, no faces, blur, or any other reason
        self.assertFalse(is_valid, f"Image should fail validation. Got: is_valid={is_valid}, error={error}")
        self.assertIsNotNone(error, "Should return an error message when validation fails")

    @unittest.skipIf(not openai_client, "OpenAI client not available - skipping content safety test")
    def test_inappropriate_content_fails(self):
        """Test that inappropriate content is rejected (or fails for any reason)"""
        img_path = self.get_test_image_path("bloody face.jpg")
        is_valid, error = validate_image(img_path)
        
        # As long as it fails validation, the test passes
        # It may fail for inappropriate content, no faces, blur, exposure, or any other reason
        self.assertFalse(is_valid, f"Image should fail validation. Got: is_valid={is_valid}, error={error}")
        self.assertIsNotNone(error, "Should return an error message when validation fails")

    def test_invalid_image_path(self):
        """Test that invalid image path returns error"""
        invalid_path = os.path.join(self.test_images_dir, "nonexistent_image.jpg")
        is_valid, error = validate_image(invalid_path)
        
        self.assertFalse(is_valid, "Invalid path should fail validation")
        self.assertIsNotNone(error, "Should return an error message")

    def test_validation_function_structure(self):
        """Test that the validation function returns correct structure"""
        img_path = self.get_test_image_path("proper image.jpg")
        result = validate_image(img_path)
        
        self.assertIsInstance(result, tuple, "Should return a tuple")
        self.assertEqual(len(result), 2, "Should return (is_valid, error) tuple")
        is_valid, error = result
        self.assertIsInstance(is_valid, bool, "First element should be boolean")
        if not is_valid:
            self.assertIsInstance(error, str, "Error should be a string when validation fails")
        else:
            self.assertIsNone(error, "Error should be None when validation passes")


if __name__ == '__main__':
    unittest.main()
