"""
Test script to verify answer extraction and comparison works correctly
"""

from mcts_parallel import MathReasoningSystemParallel, MathProblem

def test_answer_extraction():
    """Test answer extraction and comparison"""
    
    print("=" * 70)
    print("TESTING ANSWER EXTRACTION AND COMPARISON")
    print("=" * 70)
    
    # Initialize system
    system = MathReasoningSystemParallel(
        model_name="gpt-4o-mini",
        temperature=0.7,
        verbose_logging=False,
        enable_feedback=False  # Disable feedback for faster testing
    )
    
    # Test cases
    test_cases = [
        {
            "name": "Test 1: Boxed answer",
            "solution": "The answer is calculated as follows: $2x + 5 = 15$, so $2x = 10$, therefore $x = \\boxed{5}$.",
            "groundtruth": "5",
            "expected_extracted": "5",
            "expected_correct": True
        },
        {
            "name": "Test 2: Descriptive solution with answer",
            "solution": "Using the Pythagorean theorem, we find that $DE = \\sqrt{51}$.",
            "groundtruth": "\\sqrt{51}",
            "expected_extracted": None,  # No clear extraction pattern
            "expected_correct": False  # Will depend on extraction
        },
        {
            "name": "Test 3: Answer in equals pattern",
            "solution": "After solving the equation, the answer equals 2000 calories.",
            "groundtruth": "2000",
            "expected_extracted": "2000 calories",
            "expected_correct": True  # Should match via substring
        },
        {
            "name": "Test 4: Complex answer with fractions",
            "solution": "The final answer is $\\boxed{\\frac{14}{3}}$.",
            "groundtruth": "\\frac{14}{3}",
            "expected_extracted": "\\frac{14}{3}",
            "expected_correct": True
        }
    ]
    
    print("\nTesting Answer Extraction:")
    print("-" * 70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{test['name']}")
        print(f"Solution: {test['solution'][:80]}...")
        print(f"Ground Truth: {test['groundtruth']}")
        
        # Extract answer
        extracted = system._extract_answer_from_solution(test['solution'])
        print(f"Extracted: {extracted}")
        
        # Compare answers
        is_correct = system._compare_answers(extracted, test['groundtruth'])
        print(f"Comparison Result: {is_correct}")
        
        # Check if extraction worked
        if extracted:
            print(f"✅ Answer extracted successfully")
        else:
            print(f"⚠️  No answer extracted (may need better extraction)")
        
        # Check comparison
        if is_correct == test['expected_correct']:
            print(f"✅ Comparison correct")
        else:
            print(f"⚠️  Comparison: expected {test['expected_correct']}, got {is_correct}")
    
    print("\n" + "=" * 70)
    print("Testing Normalization:")
    print("-" * 70)
    
    normalization_tests = [
        ("\\frac{14}{3}", "14/3"),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3, pi/2)"),
        ("\\sqrt{51}", "sqrt(51)"),
        ("$2000$", "2000"),
    ]
    
    for original, expected_pattern in normalization_tests:
        normalized = system._normalize_answer(original)
        print(f"Original: {original}")
        print(f"Normalized: {normalized}")
        print(f"Expected pattern: {expected_pattern}")
        print()
    
    print("=" * 70)
    print("✅ Answer extraction and comparison test complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_answer_extraction()





