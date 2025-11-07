#!/usr/bin/env python3
"""
Simple Weight Ratio Testing

Test three main weight configurations on fixed 35 problems:
- 0.7/0.3 (Neural Heavy)
- 0.5/0.5 (Balanced) 
- 0.3/0.7 (LEAN Heavy)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcts_parallel import MathReasoningSystemParallel
import requests
import time
import json
from datetime import datetime

def check_lean_server():
    """Check if LEAN 4 server is running"""
    try:
        response = requests.get('http://localhost:8003/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def run_weight_test(neural_weight, symbolic_weight, test_name, problems_to_test):
    """Run test with specific weight configuration on fixed problems"""
    print(f"\n🧪 Testing {test_name}")
    print(f"   Neural: {neural_weight}, Symbolic: {symbolic_weight}")
    print(f"   Problems: {len(problems_to_test)} fixed problems")
    print("=" * 50)
    
    try:
        # Initialize system
        system = MathReasoningSystemParallel(
            model_name="gpt-4o-mini",
            temperature=0.7,
            verbose_logging=False,
            enable_feedback=True,
            feedback_type="combined",
            lean_server_url="http://localhost:8003",
            neural_weight=neural_weight,
            symbolic_weight=symbolic_weight,
            max_workers=2,
            parallel_expansions=2
        )
        
        print("✅ System initialized")
        
        # Run evaluation on the SAME fixed problems
        start_time = time.time()
        
        # Custom evaluation on fixed problems
        results = evaluate_fixed_problems(system, problems_to_test)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Extract metrics
        accuracy = results['overall_stats']['correct'] / results['overall_stats']['total']
        correct = results['overall_stats']['correct']
        total = results['overall_stats']['total']
        
        # Get feedback stats
        feedback_stats = system.get_feedback_statistics()
        
        result = {
            'test_name': test_name,
            'neural_weight': neural_weight,
            'symbolic_weight': symbolic_weight,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'time_taken': total_time,
            'feedback_calls': feedback_stats.get('total_feedback_calls', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   📊 Results: {correct}/{total} = {accuracy:.2%}")
        print(f"   ⏱️  Time: {total_time:.1f}s")
        print(f"   🔬 Feedback: {feedback_stats.get('total_feedback_calls', 0)} calls")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return {
            'test_name': test_name,
            'neural_weight': neural_weight,
            'symbolic_weight': symbolic_weight,
            'accuracy': 0.0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def evaluate_fixed_problems(system, problems):
    """Evaluate system on fixed problems"""
    results = []
    correct = 0
    total = len(problems)
    
    for i, problem in enumerate(problems):
        print(f"   Problem {i+1}/{total}: {problem['subject']} Level {problem['level']}")
        
        try:
            # Create problem object
            problem_obj = {
                'problem': problem['problem_text'],
                'answer': problem['groundtruth_answer'],
                'solution': problem['groundtruth_solution']
            }
            
            # Solve the problem
            result = system.solve_problem(
                problem=problem_obj,
                dataset_idx=problem['dataset_index']
            )
            
            # Add problem metadata
            result['dataset_index'] = problem['dataset_index']
            result['subject'] = problem['subject']
            result['level'] = problem['level']
            
            results.append(result)
            
            if result.get('is_correct', False):
                correct += 1
                
        except Exception as e:
            print(f"     ❌ Error: {e}")
            # Add error result
            error_result = {
                'dataset_index': problem['dataset_index'],
                'subject': problem['subject'],
                'level': problem['level'],
                'problem_text': problem['problem_text'],
                'is_correct': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(error_result)
    
    # Create results structure similar to evaluate_all_subjects_and_levels
    return {
        'problems': results,
        'overall_stats': {
            'correct': correct,
            'total': total
        }
    }

def load_problems_from_json(json_file):
    """Load the exact same problems from previous results for fair comparison"""
    print(f"📋 Loading problems from {json_file}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    problems = []
    for problem in data['problems']:
        problems.append({
            'dataset_index': problem['dataset_index'],
            'problem_text': problem['problem_text'],
            'subject': problem.get('subject', 'Unknown'),
            'level': problem.get('level', 1),
            'groundtruth_answer': problem.get('groundtruth_answer', ''),
            'groundtruth_solution': problem.get('groundtruth_solution', '')
        })
    
    print(f"✅ Loaded {len(problems)} problems from previous test")
    return problems

def get_fixed_problems():
    """Get a fixed set of 35 problems for fair comparison"""
    print("📋 Getting fixed set of 35 problems...")
    
    # Load dataset directly to get fixed problems
    from datasets import load_dataset
    
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    # All subjects from MATH-500
    all_subjects = ['Precalculus', 'Intermediate Algebra', 'Algebra', 'Number Theory',
                   'Prealgebra', 'Geometry', 'Counting & Probability']
    all_levels = [1, 2, 3, 4, 5]
    
    # Group problems by subject and level
    from collections import defaultdict
    problems_by_category = defaultdict(list)
    for idx, item in enumerate(dataset):
        subject = item.get('subject', 'Unknown')
        level = item.get('level', 0)
        if subject in all_subjects and level in all_levels:
            problems_by_category[(subject, level)].append((idx, item))
    
    # Select first problem from each category (same selection logic as evaluate_all_subjects_and_levels)
    selected_problems = []
    for subject in all_subjects:
        for level in all_levels:
            category_problems = problems_by_category.get((subject, level), [])
            if category_problems:
                # Take the first problem from this category
                idx, item = category_problems[0]
                selected_problems.append({
                    'dataset_index': idx,
                    'problem_text': item['problem'],
                    'subject': subject,
                    'level': level,
                    'groundtruth_answer': item['answer'],
                    'groundtruth_solution': item['solution']
                })
    
    print(f"✅ Got {len(selected_problems)} fixed problems for testing")
    return selected_problems

def main():
    """Run simple weight ratio tests on the SAME problems"""
    print("SIMPLE WEIGHT RATIO TESTING")
    print("=" * 50)
    print("Testing 3 weight configurations on the SAME 35 problems")
    print("This ensures fair comparison! 🎯")
    
    # Check LEAN server
    if not check_lean_server():
        print("❌ LEAN 4 server not running. Please start it first:")
        print("   python real_lean_server.py")
        return
    
    print("✅ LEAN 4 server is running")
    
    # Load the EXACT same problems from previous test for fair comparison
    problems_to_test = load_problems_from_json("mcts_math_results_parallel_20251019_162111.json")
    
    # Define weight configurations - Testing "All LEAN" (0.0 Neural, 1.0 Symbolic)
    weight_configs = [
        (0.0, 1.0, "All LEAN")
    ]
    
    results = []
    
    # Run tests on the SAME problems
    for neural_weight, symbolic_weight, test_name in weight_configs:
        result = run_weight_test(neural_weight, symbolic_weight, test_name, problems_to_test)
        results.append(result)
        
        # Save individual result with descriptive filename
        filename = f"{test_name.lower().replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"   💾 Saved: {filename}")
        print(f"   📊 Detailed results: mcts_math_results_parallel_*.json (created automatically)")
    
    # Analyze results
    print(f"\n📊 RESULTS COMPARISON")
    print("=" * 50)
    
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'Configuration':<15} {'Weights':<12} {'Accuracy':<10} {'Time':<8}")
    print("-" * 60)
    
    for i, result in enumerate(sorted_results):
        weights_str = f"{result['neural_weight']:.1f}/{result['symbolic_weight']:.1f}"
        time_str = f"{result.get('time_taken', 0):.1f}s" if 'time_taken' in result else "N/A"
        print(f"{i+1:<4} {result['test_name']:<15} {weights_str:<12} "
              f"{result['accuracy']:.2%} {'':<3} {time_str}")
    
    # Find best configuration
    best = sorted_results[0]
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   {best['test_name']}: {best['neural_weight']:.1f}/{best['symbolic_weight']:.1f}")
    print(f"   Accuracy: {best['accuracy']:.2%}")
    time_str = f"{best.get('time_taken', 0):.1f}s" if 'time_taken' in best else "N/A"
    print(f"   Time: {time_str}")
    
    # Calculate improvements
    if len(sorted_results) >= 2:
        best_acc = sorted_results[0]['accuracy']
        second_acc = sorted_results[1]['accuracy']
        improvement = best_acc - second_acc
        
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        print(f"   Best vs Second: {improvement:+.2%}")
        
        if improvement > 0.05:  # 5% improvement
            print("   ✅ Significant improvement over second best")
        elif improvement > 0.02:  # 2% improvement
            print("   ⚠️  Modest improvement over second best")
        else:
            print("   ⚠️  Marginal difference between configurations")
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = f"weight_comparison_{timestamp}.json"
    
    with open(combined_file, 'w') as f:
        json.dump({
            'experiment_info': {
                'timestamp': timestamp,
                'total_tests': len(results),
                'experiment_type': 'simple_weight_testing'
            },
            'results': results,
            'best_configuration': best,
            'analysis': {
                'sorted_by_accuracy': sorted_results,
                'best_accuracy': best['accuracy'],
                'best_weights': f"{best['neural_weight']:.1f}/{best['symbolic_weight']:.1f}"
            }
        }, f, indent=2)
    
    print(f"\n💾 Combined results saved to: {combined_file}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   🎯 Use {best['test_name']} configuration for full MATH-500")
    print(f"   📊 Weights: {best['neural_weight']:.1f}/{best['symbolic_weight']:.1f}")
    print(f"   🚀 Expected accuracy: {best['accuracy']:.1%} on full dataset")
    
    if best['test_name'] == "Balanced":
        print(f"   ✅ Balanced approach works best - equal neural/symbolic influence")
    elif best['test_name'] == "Neural Heavy":
        print(f"   ✅ Neural-heavy approach works best - LLM reasoning dominates")
    elif best['test_name'] == "LEAN Heavy":
        print(f"   ✅ LEAN-heavy approach works best - formal verification dominates")
    
    print(f"\n🎉 Weight testing completed!")
    print(f"📁 Check individual result files for detailed analysis")

if __name__ == "__main__":
    main()
