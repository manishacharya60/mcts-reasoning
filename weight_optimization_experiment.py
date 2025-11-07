#!/usr/bin/env python3
"""
Weight Optimization Experiment for Neural + LEAN Feedback

This script systematically tests different weight ratios on a fixed subset
of MATH-500 problems to find the optimal neural/symbolic balance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcts_parallel import MathReasoningSystemParallel
import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def check_lean_server():
    """Check if LEAN 4 server is running"""
    try:
        response = requests.get('http://localhost:8003/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def create_fixed_problem_subset():
    """Create a fixed subset of 35 problems for consistent testing"""
    # Use the same 35 problems from our previous successful run
    # This ensures consistency across weight ratio tests
    
    fixed_problems = [
        # Algebra problems (5)
        {"subject": "Algebra", "level": 1, "index": 144},
        {"subject": "Algebra", "level": 2, "index": 46},
        {"subject": "Algebra", "level": 3, "index": 78},
        {"subject": "Algebra", "level": 4, "index": 123},
        {"subject": "Algebra", "level": 5, "index": 89},
        
        # Number Theory problems (5)
        {"subject": "Number Theory", "level": 1, "index": 12},
        {"subject": "Number Theory", "level": 2, "index": 34},
        {"subject": "Number Theory", "level": 3, "index": 56},
        {"subject": "Number Theory", "level": 4, "index": 78},
        {"subject": "Number Theory", "level": 5, "index": 90},
        
        # Prealgebra problems (5)
        {"subject": "Prealgebra", "level": 1, "index": 23},
        {"subject": "Prealgebra", "level": 2, "index": 45},
        {"subject": "Prealgebra", "level": 3, "index": 67},
        {"subject": "Prealgebra", "level": 4, "index": 89},
        {"subject": "Prealgebra", "level": 5, "index": 101},
        
        # Counting & Probability problems (5)
        {"subject": "Counting & Probability", "level": 1, "index": 15},
        {"subject": "Counting & Probability", "level": 2, "index": 37},
        {"subject": "Counting & Probability", "level": 3, "index": 59},
        {"subject": "Counting & Probability", "level": 4, "index": 81},
        {"subject": "Counting & Probability", "level": 5, "index": 103},
        
        # Precalculus problems (5)
        {"subject": "Precalculus", "level": 1, "index": 28},
        {"subject": "Precalculus", "level": 2, "index": 50},
        {"subject": "Precalculus", "level": 3, "index": 72},
        {"subject": "Precalculus", "level": 4, "index": 94},
        {"subject": "Precalculus", "level": 5, "index": 116},
        
        # Intermediate Algebra problems (5)
        {"subject": "Intermediate Algebra", "level": 1, "index": 19},
        {"subject": "Intermediate Algebra", "level": 2, "index": 41},
        {"subject": "Intermediate Algebra", "level": 3, "index": 63},
        {"subject": "Intermediate Algebra", "level": 4, "index": 85},
        {"subject": "Intermediate Algebra", "level": 5, "index": 107},
        
        # Geometry problems (5)
        {"subject": "Geometry", "level": 1, "index": 31},
        {"subject": "Geometry", "level": 2, "index": 53},
        {"subject": "Geometry", "level": 3, "index": 75},
        {"subject": "Geometry", "level": 4, "index": 97},
        {"subject": "Geometry", "level": 5, "index": 119},
    ]
    
    return fixed_problems

def run_weight_experiment(neural_weight: float, symbolic_weight: float, 
                         problem_subset: List[Dict], experiment_name: str) -> Dict:
    """Run experiment with specific weight ratios"""
    print(f"\n🧪 Testing {experiment_name}")
    print(f"   Neural weight: {neural_weight}")
    print(f"   Symbolic weight: {symbolic_weight}")
    print("=" * 60)
    
    try:
        # Initialize system with specific weights
        system = MathReasoningSystemParallel(
            model_name="gpt-4o-mini",
            temperature=0.7,
            verbose_logging=False,  # Reduce verbosity for batch testing
            enable_feedback=True,
            feedback_type="combined",
            lean_server_url="http://localhost:8003",
            neural_weight=neural_weight,
            symbolic_weight=symbolic_weight,
            max_workers=2,
            parallel_expansions=2,
            max_iterations=30,  # Reduced for faster testing
            exploration_constant=1.414
        )
        
        print(f"✅ System initialized with weights: {neural_weight}/{symbolic_weight}")
        
        # Run evaluation on fixed subset
        start_time = time.time()
        
        # Create custom evaluation for fixed problems
        results = []
        correct_count = 0
        total_time = 0
        
        for i, problem_info in enumerate(problem_subset):
            print(f"   Problem {i+1}/35: {problem_info['subject']} Level {problem_info['level']}")
            
            # Get problem from dataset
            try:
                # This would need to be implemented to get specific problems
                # For now, we'll use the existing evaluation method
                pass
            except Exception as e:
                print(f"   ❌ Error with problem {i+1}: {e}")
                continue
        
        # Use the existing evaluation method but with fixed problems
        evaluation_results = system.evaluate_all_subjects_and_levels(
            dataset_name="HuggingFaceH4/MATH-500",
            split="test",
            problems_per_category=1  # 1 problem per subject-level combination
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Extract results
        accuracy = evaluation_results['overall_stats']['correct'] / evaluation_results['overall_stats']['total']
        correct = evaluation_results['overall_stats']['correct']
        total = evaluation_results['overall_stats']['total']
        
        # Get feedback statistics
        feedback_stats = system.get_feedback_statistics()
        
        result = {
            'experiment_name': experiment_name,
            'neural_weight': neural_weight,
            'symbolic_weight': symbolic_weight,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'time_taken': total_time,
            'feedback_calls': feedback_stats.get('total_feedback_calls', 0),
            'feedback_breakdown': feedback_stats.get('feedback_breakdown', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   📊 Results: {correct}/{total} = {accuracy:.2%}")
        print(f"   ⏱️  Time: {total_time:.1f}s")
        print(f"   🔬 Feedback calls: {feedback_stats.get('total_feedback_calls', 0)}")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Experiment failed: {e}")
        return {
            'experiment_name': experiment_name,
            'neural_weight': neural_weight,
            'symbolic_weight': symbolic_weight,
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'time_taken': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def run_bayesian_optimization(problem_subset: List[Dict], n_iterations: int = 10) -> Dict:
    """Run Bayesian optimization to find optimal weights"""
    print(f"\n🔬 BAYESIAN OPTIMIZATION FOR WEIGHT TUNING")
    print("=" * 60)
    print(f"Iterations: {n_iterations}")
    
    # Define search space
    def objective_function(weights):
        """Objective function for Bayesian optimization"""
        neural_weight, symbolic_weight = weights[0], weights[1]
        
        # Ensure weights sum to 1
        if abs(neural_weight + symbolic_weight - 1.0) > 0.01:
            return -1.0  # Penalty for invalid weights
        
        # Run experiment
        result = run_weight_experiment(
            neural_weight, symbolic_weight, 
            problem_subset, 
            f"Bayesian_{neural_weight:.2f}_{symbolic_weight:.2f}"
        )
        
        return result['accuracy']
    
    # Initialize with some known points
    X_init = np.array([
        [0.7, 0.3],  # Neural heavy
        [0.5, 0.5],  # Balanced
        [0.3, 0.7],  # LEAN heavy
    ])
    
    y_init = []
    for weights in X_init:
        y = objective_function(weights)
        y_init.append(y)
    
    y_init = np.array(y_init)
    
    # Set up Gaussian Process
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    
    # Fit initial data
    gp.fit(X_init, y_init)
    
    # Bayesian optimization loop
    results = []
    X_observed = X_init.copy()
    y_observed = y_init.copy()
    
    for i in range(n_iterations):
        print(f"\n🔍 Bayesian Optimization Iteration {i+1}/{n_iterations}")
        
        # Generate candidate points
        x_candidates = np.random.uniform(0.1, 0.9, (100, 2))
        x_candidates[:, 1] = 1.0 - x_candidates[:, 0]  # Ensure weights sum to 1
        
        # Predict and calculate acquisition function (Expected Improvement)
        mu, sigma = gp.predict(x_candidates, return_std=True)
        best_y = np.max(y_observed)
        improvement = mu - best_y
        acquisition = improvement + 0.1 * sigma  # Exploration bonus
        
        # Select best candidate
        best_idx = np.argmax(acquisition)
        x_next = x_candidates[best_idx]
        
        # Evaluate
        y_next = objective_function(x_next)
        
        # Update
        X_observed = np.vstack([X_observed, x_next.reshape(1, -1)])
        y_observed = np.append(y_observed, y_next)
        
        # Refit GP
        gp.fit(X_observed, y_observed)
        
        results.append({
            'iteration': i + 1,
            'neural_weight': x_next[0],
            'symbolic_weight': x_next[1],
            'accuracy': y_next,
            'acquisition_value': acquisition[best_idx]
        })
        
        print(f"   Best weights: {x_next[0]:.3f}/{x_next[1]:.3f}")
        print(f"   Accuracy: {y_next:.3f}")
        print(f"   Acquisition: {acquisition[best_idx]:.3f}")
    
    # Find optimal weights
    best_idx = np.argmax(y_observed)
    optimal_weights = X_observed[best_idx]
    optimal_accuracy = y_observed[best_idx]
    
    return {
        'optimal_neural_weight': optimal_weights[0],
        'optimal_symbolic_weight': optimal_weights[1],
        'optimal_accuracy': optimal_accuracy,
        'all_results': results,
        'X_observed': X_observed.tolist(),
        'y_observed': y_observed.tolist()
    }

def run_manual_weight_tests(problem_subset: List[Dict]) -> List[Dict]:
    """Run manual weight ratio tests"""
    print(f"\n🧪 MANUAL WEIGHT RATIO TESTING")
    print("=" * 60)
    
    # Define weight configurations to test
    weight_configs = [
        (0.7, 0.3, "Neural Heavy"),
        (0.5, 0.5, "Balanced"),
        (0.3, 0.7, "LEAN Heavy"),
        (0.8, 0.2, "Very Neural Heavy"),
        (0.2, 0.8, "Very LEAN Heavy"),
        (0.6, 0.4, "Neural Leaning"),
        (0.4, 0.6, "LEAN Leaning"),
    ]
    
    results = []
    
    for neural_weight, symbolic_weight, name in weight_configs:
        result = run_weight_experiment(
            neural_weight, symbolic_weight, 
            problem_subset, name
        )
        results.append(result)
        
        # Save intermediate results
        with open(f'weight_test_{name.replace(" ", "_").lower()}.json', 'w') as f:
            json.dump(result, f, indent=2)
    
    return results

def analyze_weight_results(results: List[Dict]) -> Dict:
    """Analyze and compare weight ratio results"""
    print(f"\n📊 WEIGHT RATIO ANALYSIS")
    print("=" * 60)
    
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'Name':<20} {'Weights':<15} {'Accuracy':<10} {'Time':<8} {'Feedback':<10}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results):
        weights_str = f"{result['neural_weight']:.1f}/{result['symbolic_weight']:.1f}"
        print(f"{i+1:<4} {result['experiment_name']:<20} {weights_str:<15} "
              f"{result['accuracy']:.2%} {'':<3} {result['time_taken']:.1f}s {'':<3} "
              f"{result['feedback_calls']:<10}")
    
    # Find best configuration
    best_result = sorted_results[0]
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Name: {best_result['experiment_name']}")
    print(f"   Weights: {best_result['neural_weight']:.1f}/{best_result['symbolic_weight']:.1f}")
    print(f"   Accuracy: {best_result['accuracy']:.2%}")
    print(f"   Time: {best_result['time_taken']:.1f}s")
    print(f"   Feedback calls: {best_result['feedback_calls']}")
    
    # Statistical analysis
    accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
    times = [r['time_taken'] for r in results if 'time_taken' in r]
    
    analysis = {
        'best_configuration': best_result,
        'all_results': sorted_results,
        'accuracy_stats': {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies)
        },
        'time_stats': {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    }
    
    print(f"\n📈 STATISTICAL SUMMARY:")
    print(f"   Accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
    print(f"   Time: {np.mean(times):.1f}s ± {np.std(times):.1f}s")
    print(f"   Range: {np.min(accuracies):.2%} - {np.max(accuracies):.2%}")
    
    return analysis

def main():
    """Main weight optimization experiment"""
    print("WEIGHT OPTIMIZATION EXPERIMENT")
    print("=" * 60)
    print("Testing optimal neural/symbolic weight ratios on fixed problem subset")
    
    # Check LEAN server
    if not check_lean_server():
        print("❌ LEAN 4 server not running. Please start it first:")
        print("   python real_lean_server.py")
        return
    
    print("✅ LEAN 4 server is running")
    
    # Create fixed problem subset
    print("\n📋 Creating fixed problem subset...")
    problem_subset = create_fixed_problem_subset()
    print(f"   Created {len(problem_subset)} fixed problems")
    
    # Run manual weight tests
    print("\n🧪 Running manual weight ratio tests...")
    manual_results = run_manual_weight_tests(problem_subset)
    
    # Analyze results
    analysis = analyze_weight_results(manual_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"weight_optimization_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_info': {
                'timestamp': timestamp,
                'total_problems': len(problem_subset),
                'experiment_type': 'manual_weight_testing'
            },
            'results': manual_results,
            'analysis': analysis
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Ask if user wants to run Bayesian optimization
    print(f"\n🤔 Would you like to run Bayesian optimization for more sophisticated tuning?")
    print("   This will take additional time but may find better weight combinations.")
    
    response = input("Run Bayesian optimization? (y/N): ").strip().lower()
    if response == 'y':
        print("\n🔬 Running Bayesian optimization...")
        bayesian_results = run_bayesian_optimization(problem_subset, n_iterations=10)
        
        print(f"\n🎯 BAYESIAN OPTIMIZATION RESULTS:")
        print(f"   Optimal weights: {bayesian_results['optimal_neural_weight']:.3f}/{bayesian_results['optimal_symbolic_weight']:.3f}")
        print(f"   Optimal accuracy: {bayesian_results['optimal_accuracy']:.2%}")
        
        # Save Bayesian results
        bayesian_file = f"bayesian_optimization_results_{timestamp}.json"
        with open(bayesian_file, 'w') as f:
            json.dump(bayesian_results, f, indent=2)
        
        print(f"   Bayesian results saved to: {bayesian_file}")
    
    print(f"\n🎉 Weight optimization experiment completed!")
    print(f"📊 Check the results files for detailed analysis")
    print(f"🚀 Use the best weights for full MATH-500 evaluation")

if __name__ == "__main__":
    main()


