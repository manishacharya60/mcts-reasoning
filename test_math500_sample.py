#!/usr/bin/env python3
"""
Test script for feedback-aware MCTS on MATH-500 dataset sample

This script runs a small sample of MATH-500 problems to validate the feedback system
before running on the full dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcts_parallel import MathReasoningSystemParallel

def test_math500_sample():
    """Test with a small sample of MATH-500 problems"""
    print("=" * 80)
    print("TESTING FEEDBACK-AWARE MCTS ON MATH-500 SAMPLE")
    print("=" * 80)
    
    # Initialize system with neural feedback
    system = MathReasoningSystemParallel(
        model_name="gpt-4o-mini",
        temperature=0.7,
        verbose_logging=True,
        enable_feedback=True,
        feedback_type="neural",
        max_workers=2,  # Reduced for testing
        parallel_expansions=2  # Reduced for testing
    )
    
    # Test with a small sample from MATH-500
    print("\n🔍 Testing on MATH-500 sample (5 problems)...")
    
    try:
        results = system.evaluate_all_subjects_and_levels(
            dataset_name="HuggingFaceH4/MATH-500",
            split="test",
            problems_per_category=1  # Just 1 problem per subject-level combination
        )
        
        print(f"\n✅ Sample evaluation completed!")
        print(f"📊 Results saved to: {results['results_file']}")
        print(f"📈 Total problems evaluated: {results['num_problems_evaluated']}")
        print(f"🎯 Overall accuracy: {results['overall_stats']['correct']}/{results['overall_stats']['total']} = "
              f"{results['overall_stats']['correct']/results['overall_stats']['total']:.2%}")
        
        # Show feedback statistics
        feedback_stats = system.get_feedback_statistics()
        print(f"\n🔬 Feedback Statistics:")
        print(f"   Total feedback calls: {feedback_stats.get('total_feedback_calls', 0)}")
        print(f"   Feedback breakdown: {feedback_stats.get('feedback_breakdown', {})}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during MATH-500 sample evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_different_feedback_modes():
    """Test different feedback configurations"""
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT FEEDBACK CONFIGURATIONS")
    print("=" * 80)
    
    configurations = [
        {"name": "No Feedback", "enable_feedback": False},
        {"name": "Neural Feedback", "enable_feedback": True, "feedback_type": "neural"},
        {"name": "Combined Feedback", "enable_feedback": True, "feedback_type": "combined", 
         "neural_weight": 0.7, "symbolic_weight": 0.3}
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n🧪 Testing: {config['name']}")
        
        try:
            system = MathReasoningSystemParallel(
                model_name="gpt-4o-mini",
                temperature=0.7,
                verbose_logging=False,  # Reduce output for comparison
                max_workers=2,
                parallel_expansions=2,
                **{k: v for k, v in config.items() if k != 'name'}
            )
            
            # Test with a simple problem
            problem = "Solve for x: 2x + 3 = 11"
            result = system.solve_problem(problem, 1, 1)
            
            results[config['name']] = {
                'solution': result.get('solution'),
                'is_correct': result.get('is_correct'),
                'time_taken': result.get('time_taken'),
                'feedback_enabled': result.get('feedback_enabled'),
                'feedback_stats': system.get_feedback_statistics()
            }
            
            print(f"   Solution: {result.get('solution')}")
            print(f"   Correct: {result.get('is_correct')}")
            print(f"   Time: {result.get('time_taken', 0):.2f}s")
            print(f"   Feedback: {result.get('feedback_enabled')}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[config['name']] = {'error': str(e)}
    
    return results

def main():
    """Run comprehensive testing"""
    print("FEEDBACK-AWARE MCTS MATH-500 VALIDATION")
    print("=" * 80)
    
    # Test 1: Different feedback modes
    print("\n1️⃣ Testing different feedback configurations...")
    config_results = test_different_feedback_modes()
    
    # Test 2: MATH-500 sample
    print("\n2️⃣ Testing on MATH-500 sample...")
    sample_results = test_math500_sample()
    
    # Summary
    print("\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)
    
    print("\n📊 Configuration Comparison:")
    for name, result in config_results.items():
        if 'error' in result:
            print(f"   {name}: ❌ {result['error']}")
        else:
            print(f"   {name}: ✅ Solution={result['solution']}, Correct={result['is_correct']}")
    
    if sample_results:
        print(f"\n📈 MATH-500 Sample Results:")
        print(f"   Problems evaluated: {sample_results['num_problems_evaluated']}")
        print(f"   Overall accuracy: {sample_results['overall_stats']['correct']}/{sample_results['overall_stats']['total']}")
        print(f"   Results file: {sample_results['results_file']}")
    
    print(f"\n✅ All tests completed!")
    print(f"\n🚀 Ready for full MATH-500 evaluation!")

if __name__ == "__main__":
    main()
