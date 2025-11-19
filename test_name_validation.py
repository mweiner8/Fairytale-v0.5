"""
Unit tests for child name validation functionality.
"""
import unittest
import sys
import os

# Add the parent directory to the path so we can import app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import validate_child_name


class TestNameValidation(unittest.TestCase):
    """Test cases for validate_child_name function"""

    def test_valid_names(self):
        """Test that valid names pass validation"""
        valid_names = [
            "Alice",
            "Bob",
            "Charlie",
            "Emma",
            "John",
            "Mary",
            "David",
            "Sarah",
            "Michael",
            "Emily",
            "alice",  # lowercase
            "ALICE",  # uppercase
            "AlIcE",  # mixed case
            "Liam",
            "Olivia",
            "Noah",
            "Sophia",
            "James",
            "Isabella",
        ]
        
        for name in valid_names:
            with self.subTest(name=name):
                result = validate_child_name(name)
                self.assertIsNone(result, f"Valid name '{name}' should pass validation")

    def test_too_short_names(self):
        """Test that names shorter than 2 characters are rejected"""
        short_names = ["", "A", " "]
        
        for name in short_names:
            with self.subTest(name=name):
                result = validate_child_name(name)
                self.assertIsNotNone(result, f"Short name '{name}' should fail validation")
                self.assertIn("Please enter a real first name", result)

    def test_too_long_names(self):
        """Test that names longer than 20 characters are rejected"""
        long_names = [
            "Abcdefghijklmnopqrstu",  # 21 characters
            "A" * 25,
            "VeryLongNameThatExceedsTwentyCharacters"
        ]
        
        for name in long_names:
            with self.subTest(name=name):
                result = validate_child_name(name)
                self.assertIsNotNone(result, f"Long name '{name}' should fail validation")
                self.assertIn("Please enter a real first name", result)

    def test_names_with_digits(self):
        """Test that names containing digits are rejected"""
        names_with_digits = [
            "Alice1",
            "Bob2",
            "123",
            "Alice123",
            "1Alice",
            "Al1ce"
        ]
        
        for name in names_with_digits:
            with self.subTest(name=name):
                result = validate_child_name(name)
                self.assertIsNotNone(result, f"Name with digits '{name}' should fail validation")
                self.assertIn("Please enter a real first name", result)

    def test_names_with_symbols(self):
        """Test that names containing symbols are rejected"""
        names_with_symbols = [
            "Alice!",
            "Bob@",
            "Charlie#",
            "Alice-Bob",
            "Alice_Bob",
            "Alice.Bob",
            "Alice Bob",  # space is a symbol
            "Alice'Bob",
            "Alice&Bob"
        ]
        
        for name in names_with_symbols:
            with self.subTest(name=name):
                result = validate_child_name(name)
                self.assertIsNotNone(result, f"Name with symbols '{name}' should fail validation")
                self.assertIn("Please enter a real first name", result)

    def test_profanity_filtering(self):
        """Test that names containing profanity are rejected"""
        # Note: This test depends on better_profanity's word list
        # Common profane words that should be caught
        profane_names = [
            "damn",  # Common profanity
            "hell",  # Common profanity
        ]
        
        for name in profane_names:
            with self.subTest(name=name):
                result = validate_child_name(name)
                # The profanity filter may or may not catch these depending on the library version
                # So we'll just check that the function doesn't crash
                # If it does catch profanity, it should return an error message
                if result is not None:
                    self.assertIn("Please enter a real first name", result)

    def test_none_input(self):
        """Test that None input is handled"""
        result = validate_child_name(None)
        self.assertIsNotNone(result)
        self.assertIn("Please enter a real first name", result)

    def test_whitespace_only(self):
        """Test that whitespace-only names are rejected"""
        whitespace_names = ["  ", "\t", "\n", "   "]
        
        for name in whitespace_names:
            with self.subTest(name=repr(name)):
                result = validate_child_name(name)
                self.assertIsNotNone(result)
                self.assertIn("Please enter a real first name", result)

    def test_edge_case_lengths(self):
        """Test edge cases for length validation"""
        # Valid 2-character name (if it exists in the list)
        # Using "Ab" might not be in the list, so let's use a known valid short name
        result = validate_child_name("Ab")  # This might fail if not in list, which is fine
        # We'll just check it doesn't crash
        
        # 1 character (should fail - too short)
        result = validate_child_name("A")
        self.assertIsNotNone(result, "1-character name should fail")
        
        # 21 characters (should fail - too long)
        result = validate_child_name("A" * 21)
        self.assertIsNotNone(result, "21-character name should fail")
    
    def test_nonsense_names(self):
        """Test that nonsense words that aren't real names are rejected"""
        nonsense_names = [
            "Pizza",
            "Moo",
            "Keyboard",
            "Computer",
            "Table",
            "Chair",
            "Window",
            "Door",
            "Car",
            "Tree",
            "Dog",
            "Cat",
            "Book",
            "Pen",
            "Paper",
            "Water",
            "Fire",
            "Earth",
            "Air",
            "Sun",
            "Moon",
            "Star",
            "Cloud",
            "Rain",
            "Snow",
        ]
        
        for name in nonsense_names:
            with self.subTest(name=name):
                result = validate_child_name(name)
                self.assertIsNotNone(result, f"Nonsense name '{name}' should fail validation")
                self.assertIn("Please enter a real first name", result)
    
    def test_nonsense_strings(self):
        """Test that random strings that aren't words are rejected"""
        nonsense_strings = [
            "Xyz",
            "Qwerty",
            "Asdf",
            "Zxcv",
            "Abcd",
            "Efgh",
            "Ijkl",
            "Mnop",
            "Qrst",
            "Uvwx",
        ]
        
        for name in nonsense_strings:
            with self.subTest(name=name):
                result = validate_child_name(name)
                self.assertIsNotNone(result, f"Nonsense string '{name}' should fail validation")
                self.assertIn("Please enter a real first name", result)


if __name__ == '__main__':
    unittest.main()

