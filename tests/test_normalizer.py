import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxforge.normalizer import normalize

def test_currency_dollars():
    result = normalize("The company raised $1.5M last year.")
    assert "million" in result, f"Expected 'million' in: {result}"
    assert "dollar" in result, f"Expected 'dollar' in: {result}"
    print(f"  PASS: {result}")

def test_plain_number():
    result = normalize("There are 42 items in the list.")
    assert "forty" in result.lower(), f"Expected 'forty-two' in: {result}"
    print(f"  PASS: {result}")

def test_abbreviation_doctor():
    result = normalize("Dr. Smith will see you now.")
    assert "doctor" in result.lower(), f"Expected 'doctor' in: {result}"
    print(f"  PASS: {result}")

def test_abbreviation_versus():
    result = normalize("It's Python vs. JavaScript.")
    assert "versus" in result.lower(), f"Expected 'versus' in: {result}"
    print(f"  PASS: {result}")

def test_empty_input():
    try:
        normalize("")
        print("  FAIL: Should have raised ValueError")
    except ValueError:
        print("  PASS: Empty input raised ValueError correctly")

def test_special_chars():
    result = normalize("Hello @world! #test ^caret")
    assert "@" not in result, f"@ should be removed: {result}"
    assert "#" not in result, f"# should be removed: {result}"
    print(f"  PASS: {result}")
    
def test_year():
    result = normalize("The project started in 2023 and ends in 2025.")
    assert "twenty twenty-three" in result.lower(), f"Expected year spoken: {result}"
    assert "twenty twenty-five" in result.lower(), f"Expected year spoken: {result}"
    print(f"  PASS: {result}")

if __name__ == "__main__":
    print("Running normalizer tests...\n")
    test_currency_dollars()
    test_plain_number()
    test_abbreviation_doctor()
    test_abbreviation_versus()
    test_empty_input()
    test_special_chars()
    test_year()
    print("\nAll normalizer tests passed.")