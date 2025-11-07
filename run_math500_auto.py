#!/usr/bin/env python3
"""
Automated MATH-500 Evaluation with Combined System

This script automatically runs the MATH-500 evaluation with the combined
neural + LEAN system without user confirmation.
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
        return response.status_code == 200
    except:
        return False

def run_math500_evaluation():
    """Run MATH-500 evaluation with combined system"""
    print("=" * 80)
    print("MATH-500 EVALUATION WITH COMBINED NEURAL + LEAN SYSTEM")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check LEAN server
    if not check_lean_server():
        print("❌ LEAN 4 server not running. Please start it first:")
        print("   python real_lean_server.py")
        return None
    
    print("✅ LEAN 4 server is running")
    
    try:
        # Initialize system
        print("\n🚀 Initializing MCTS system with combined feedback...")
        system = MathReasoningSystemParallel(
            model_name="gpt-4o-mini",
            temperature=0.7,
            verbose_logging=True,
            enable_feedback=True,
            feedback_type="combined",
            lean_server_url="http://localhost:8003",
            neural_weight=0.7,
            symbolic_weight=0.3,
            max_workers=4,
            parallel_expansions=3,
            max_iterations=50,
            exploration_constant=1.414
        )
        
        print("✅ System initialized")
        
        # Run evaluation
        print("\n🔍 Starting MATH-500 evaluation...")
        start_time = time.time()
        
        results = system.evaluate_all_subjects_and_levels(
            dataset_name="HuggingFaceH4/MATH-500",
            split="test",
            problems_per_category=None  # Full dataset
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Display results
        print(f"\n✅ EVALUATION COMPLETED!")
        print("=" * 50)
        print(f"📊 Results: {results['results_file']}")
        print(f"📈 Problems: {results['num_problems_evaluated']}")
        print(f"🎯 Accuracy: {results['overall_stats']['correct']}/{results['overall_stats']['total']} = "
              f"{results['overall_stats']['correct']/results['overall_stats']['total']:.2%}")
        print(f"⏱️  Time: {total_time/3600:.2f} hours")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_math500_evaluation()


