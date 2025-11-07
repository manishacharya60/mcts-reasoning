#!/usr/bin/env python3
"""
Run Full MATH-500 Dataset with Combined Neural + LEAN System

This script runs the complete MATH-500 dataset using the combined feedback system
with real LEAN 4 integration for neuro-symbolic reasoning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcts_parallel import MathReasoningSystemParallel
import requests
import time
from datetime import datetime

def check_lean_server():
    """Check if LEAN 4 server is running"""
    try:
        response = requests.get('http://localhost:8003/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ LEAN 4 Server Status: {data.get('status', 'Unknown')}")
            print(f"   LEAN Available: {data.get('lean_available', False)}")
            print(f"   Version: {data.get('version', 'Unknown')}")
            return True
        else:
            print(f"❌ LEAN 4 Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ LEAN 4 Server connection failed: {e}")
        print("   Please start the LEAN 4 server first:")
        print("   python real_lean_server.py")
        return False

def run_full_math500_evaluation():
    """Run full MATH-500 evaluation with combined system"""
    print("=" * 80)
    print("FULL MATH-500 EVALUATION WITH COMBINED NEURAL + LEAN SYSTEM")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check LEAN server first
    if not check_lean_server():
        print("\n❌ Cannot proceed without LEAN 4 server")
        return None
    
    try:
        # Initialize system with combined feedback
        print(f"\n🚀 Initializing MCTS system with combined feedback...")
        system = MathReasoningSystemParallel(
            model_name="gpt-4o-mini",
            temperature=0.7,
            verbose_logging=True,
            enable_feedback=True,
            feedback_type="combined",  # Neural + LEAN
            lean_server_url="http://localhost:8003",
            neural_weight=0.7,        # 70% neural feedback
            symbolic_weight=0.3,      # 30% LEAN feedback
            max_workers=4,            # Increased for full dataset
            parallel_expansions=3,    # Increased for better exploration
            max_iterations=50,        # Standard iterations
            exploration_constant=1.414 # Standard UCB exploration
        )
        
        print("✅ System initialized with combined feedback")
        print(f"   Neural weight: 70%")
        print(f"   Symbolic weight: 30%")
        print(f"   LEAN server: http://localhost:8003")
        print(f"   Max workers: 4")
        print(f"   Parallel expansions: 3")
        
        # Run full MATH-500 evaluation
        print(f"\n🔍 Starting full MATH-500 evaluation...")
        print(f"   Dataset: HuggingFaceH4/MATH-500")
        print(f"   Split: test")
        print(f"   Problems per category: ALL (full dataset)")
        
        start_time = time.time()
        
        results = system.evaluate_all_subjects_and_levels(
            dataset_name="HuggingFaceH4/MATH-500",
            split="test",
            problems_per_category=None  # None = all problems (full dataset)
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Display results
        print(f"\n✅ FULL MATH-500 EVALUATION COMPLETED!")
        print("=" * 60)
        print(f"📊 Results saved to: {results['results_file']}")
        print(f"📈 Total problems evaluated: {results['num_problems_evaluated']}")
        print(f"🎯 Overall accuracy: {results['overall_stats']['correct']}/{results['overall_stats']['total']} = "
              f"{results['overall_stats']['correct']/results['overall_stats']['total']:.2%}")
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"⏱️  Average time per problem: {total_time/results['num_problems_evaluated']:.2f} seconds")
        
        # Show subject-wise performance
        print(f"\n📈 SUBJECT-WISE PERFORMANCE:")
        print("-" * 40)
        for subject, stats in results['subject_stats'].items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{subject:<20} {stats['correct']:>3}/{stats['total']:<3} = {accuracy:.2%}")
        
        # Show difficulty-wise performance
        print(f"\n📊 DIFFICULTY-WISE PERFORMANCE:")
        print("-" * 30)
        for level, stats in results['level_stats'].items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"Level {level:<2} {stats['correct']:>3}/{stats['total']:<3} = {accuracy:.2%}")
        
        # Show feedback statistics
        feedback_stats = system.get_feedback_statistics()
        print(f"\n🔬 FEEDBACK SYSTEM STATISTICS:")
        print("-" * 35)
        print(f"Total feedback calls: {feedback_stats.get('total_feedback_calls', 0)}")
        print(f"Feedback breakdown: {feedback_stats.get('feedback_breakdown', {})}")
        
        # Show LEAN-specific statistics
        if 'feedback_combined' in feedback_stats.get('feedback_breakdown', {}):
            combined_calls = feedback_stats['feedback_breakdown']['feedback_combined']
            print(f"Combined feedback calls: {combined_calls}")
            print(f"Real LEAN 4 integration: {'✅ Active' if combined_calls > 0 else '❌ Not used'}")
        
        # Performance insights
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        print("-" * 25)
        avg_time_per_problem = total_time / results['num_problems_evaluated']
        if avg_time_per_problem < 60:
            print("✅ Fast processing - system is efficient")
        elif avg_time_per_problem < 120:
            print("⚠️  Moderate processing time - consider optimization")
        else:
            print("❌ Slow processing - may need optimization")
        
        # Compare with expected performance
        expected_accuracy = 0.75  # Based on sample results
        actual_accuracy = results['overall_stats']['correct'] / results['overall_stats']['total']
        
        if actual_accuracy >= expected_accuracy:
            print(f"✅ Performance meets expectations ({actual_accuracy:.2%} >= {expected_accuracy:.2%})")
        else:
            print(f"⚠️  Performance below expectations ({actual_accuracy:.2%} < {expected_accuracy:.2%})")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during full MATH-500 evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution function"""
    print("FULL MATH-500 EVALUATION WITH COMBINED SYSTEM")
    print("=" * 60)
    print("This will run the complete MATH-500 dataset using the combined")
    print("neural + LEAN feedback system for neuro-symbolic reasoning.")
    print()
    
    # Confirm before proceeding
    print("⚠️  WARNING: This will take several hours to complete!")
    print("   Estimated time: 4-8 hours for full dataset")
    print("   Problems: ~500 mathematical problems")
    print("   Resources: High CPU/memory usage")
    print()
    
    response = input("Do you want to proceed? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Evaluation cancelled by user")
        return
    
    # Run evaluation
    results = run_full_math500_evaluation()
    
    if results:
        print(f"\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"📁 Results file: {results['results_file']}")
        print(f"📊 You can analyze the results using: python analyze_results.py")
    else:
        print(f"\n❌ EVALUATION FAILED!")
        print(f"   Check the error messages above")
        print(f"   Ensure LEAN 4 server is running: python real_lean_server.py")

if __name__ == "__main__":
    main()
