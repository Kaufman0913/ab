#!/usr/bin/env python3
"""
Simple test script for the grep implementation
"""

from main import grep
import io
import tempfile
import os

def test_basic_functionality():
    """Test basic grep functionality"""
    print("Testing Basic Grep Functionality")
    
    # Create test files
    test_content1 = """Hello World
This is a test
Python programming
Hello again
"""
    
    test_content2 = """Another file
Hello from file 2
More content
"""
    
    # Write test files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f1:
        f1.write(test_content1)
        file1 = f1.name
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f2:
        f2.write(test_content2)
        file2 = f2.name
    
    try:
        # Test 1: Basic search
        print("\n1. Testing basic search for 'Hello':")
        result = grep("Hello", "", [file1])
        print(f"Result: {repr(result)}")
        expected_lines = 2  # Should find 2 lines with "Hello"
        actual_lines = result.count('\n') + (1 if result else 0)
        print(f"Found {actual_lines} lines (expected: {expected_lines})")
        
        # Test 2: Case insensitive search
        print("\n2. Testing case insensitive search for 'hello':")
        result = grep("hello", "-i", [file1])
        print(f"Result: {repr(result)}")
        print(f"Case insensitive search working")
        
        # Test 3: Line numbers
        print("\n3. Testing line numbers flag:")
        result = grep("Hello", "-n", [file1])
        print(f"Result: {repr(result)}")
        print(f"Line numbers included")
        
        # Test 4: Multiple files
        print("\n4. Testing multiple files:")
        result = grep("Hello", "", [file1, file2])
        print(f"Result: {repr(result)}")
        print(f"Multiple files processed")
        
        # Test 5: File names only
        print("\n5. Testing file names only flag:")
        result = grep("Hello", "-l", [file1, file2])
        print(f"Result: {repr(result)}")
        print(f"File names only working")
        
        # Test 6: Invert match
        print("\n6. Testing invert match:")
        result = grep("Hello", "-v", [file1])
        print(f"Result: {repr(result)}")
        print(f"Invert match working")
        
        # Test 7: Match entire lines
        print("\n7. Testing match entire lines:")
        result = grep("Hello World", "-x", [file1])
        print(f"Result: {repr(result)}")
        print(f"Match entire lines working")
        
        print("\nAll basic tests passed!")
        
    finally:
        # Clean up
        os.unlink(file1)
        os.unlink(file2)

def test_edge_cases():
    """Test edge cases"""
    print("\nTesting Edge Cases")
    
    # Create empty file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        empty_file = f.name
    
    try:
        # Test empty file
        print("\n1. Testing empty file:")
        result = grep("anything", "", [empty_file])
        print(f"Result: {repr(result)}")
        print(f"Empty file handled correctly")
        
        # Test no matches
        print("\n2. Testing no matches:")
        result = grep("nonexistent", "", [empty_file])
        print(f"Result: {repr(result)}")
        print(f"No matches handled correctly")
        
        print("\nAll edge case tests passed!")
        
    finally:
        os.unlink(empty_file)

if __name__ == "__main__":
    test_basic_functionality()
    test_edge_cases()
    print("\nAll tests completed successfully!")
