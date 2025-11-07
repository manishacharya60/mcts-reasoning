"""
Test Three-Agent System on Fixed Problem Set

This script loads the exact same problems from previous test results
using dataset_index, then runs the three-agent system on them for fair comparison.
"""

import json
import os
from datetime import datetime
from mcts_parallel import MathReasoningSystemParallel, MathProblem
from datasets import load_dataset

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

def load_dataset_and_get_problems(problems_list):
    """Load actual problem data from MATH-500 dataset using dataset_index"""
    print("📋 Loading actual problem data from MATH-500 dataset...")
    
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    actual_problems = []
    for problem_info in problems_list:
        idx = problem_info['dataset_index']
        if idx < len(dataset):
            item = dataset[idx]
            problem = MathProblem.from_dataset(item)
            actual_problems.append(problem)
        else:
            print(f"⚠️  Warning: dataset_index {idx} out of range, skipping")
    
    print(f"✅ Loaded {len(actual_problems)} actual problems from dataset")
    return actual_problems

def evaluate_fixed_problems(problems_to_test, system):
    """Evaluate system on fixed set of problems"""
    print(f"\n{'='*70}")
    print(f"EVALUATING THREE-AGENT SYSTEM ON {len(problems_to_test)} PROBLEMS")
    print(f"{'='*70}\n")
    
    results = []
    correct = 0
    total = len(problems_to_test)
    
    for i, problem in enumerate(problems_to_test, 1):
        print(f"\n{'='*70}")
        print(f"Problem {i}/{total}: {problem.subject} Level {problem.level}")
        print(f"{'='*70}")
        print(f"Problem: {problem.problem_text[:150]}...")
        
        try:
            result = system.solve_problem(problem)
            
            is_correct = result.get('is_correct', False)
            if is_correct:
                correct += 1
            
            print(f"\n✅ Result: {result.get('predicted_answer', 'N/A')}")
            print(f"Correct: {is_correct}")
            print(f"Time taken: {result.get('time_taken', 0):.2f} seconds")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            
            # Store result with dataset index
            result['dataset_index'] = problems_to_test.index(problem) if problem in problems_to_test else i-1
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error solving problem: {e}")
            results.append({
                'dataset_index': i-1,
                'error': str(e),
                'is_correct': False
            })
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    
    return results, accuracy, correct, total

def main():
    """Run three-agent system on fixed problem set"""
    print("=" * 70)
    print("THREE-AGENT SYSTEM TEST ON FIXED PROBLEM SET")
    print("=" * 70)
    print("\nThis test uses the EXACT same problems from previous combined feedback test")
    print("for fair comparison.\n")
    
    # Check if LEAN server is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✅ LEAN 4 server is running\n")
        else:
            print("⚠️  LEAN 4 server returned non-200 status")
            print("   Results may be affected. Continue anyway? (server might still work)\n")
    except Exception as e:
        print(f"⚠️  LEAN 4 server not reachable: {e}")
        print("   Please start the LEAN server with: python real_lean_server.py")
        print("   Exiting...")
        return
    
    # Load problems from previous results
    json_file = "mcts_math_results_parallel_20251019_151842.json"
    if not os.path.exists(json_file):
        print(f"❌ Error: {json_file} not found!")
        print("   Please ensure the previous results file is in the current directory.")
        return
    
    problems_list = load_problems_from_json(json_file)
    
    # Load actual problem data from dataset
    actual_problems = load_dataset_and_get_problems(problems_list)
    
    if not actual_problems:
        print("❌ No problems loaded. Exiting.")
        return
    
    # Initialize three-agent system
    print("\n" + "=" * 70)
    print("INITIALIZING THREE-AGENT SYSTEM")
    print("=" * 70)
    system = MathReasoningSystemParallel(
        model_name="gpt-4o-mini",
        temperature=0.7,
        verbose_logging=True,
        max_workers=2,
        parallel_expansions=2,
        enable_feedback=True,
        feedback_type="three_agent",  # Use three-agent system
        lean_server_url="http://localhost:8000"
    )
    
    # Set max_iterations for testing (same as previous test: 50)
    system.solver.max_iterations = 50
    
    print("\n" + "=" * 70)
    print("SYSTEM CONFIGURATION")
    print("=" * 70)
    print(f"Model: gpt-4o-mini")
    print(f"Feedback Type: three_agent")
    print(f"Max Iterations: 50")
    print(f"Parallel Expansions: 2")
    print(f"Max Workers: 2")
    print("=" * 70 + "\n")
    
    # Evaluate on fixed problems
    results, accuracy, correct, total = evaluate_fixed_problems(actual_problems, system)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"three_agent_results_{timestamp}.json"
    
    output_data = {
        "evaluation_date": datetime.now().isoformat(),
        "total_problems": total,
        "correct": correct,
        "accuracy": accuracy,
        "model_config": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "max_iterations": 50,
            "exploration_constant": 1.414,
            "max_depth": 15,
            "parallel_expansions": 2,
            "max_workers": 2,
            "feedback_type": "three_agent"
        },
        "problems": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Compare with previous results
    print("\n" + "=" * 70)
    print("COMPARISON WITH PREVIOUS RESULTS")
    print("=" * 70)
    
    # Load previous results for comparison
    with open(json_file, 'r') as f:
        previous_data = json.load(f)
    
    previous_correct = sum(1 for p in previous_data['problems'] if p.get('is_correct', False))
    previous_total = len(previous_data['problems'])
    previous_accuracy = (previous_correct / previous_total) * 100 if previous_total > 0 else 0
    
    print(f"\nPrevious System (Combined Feedback):")
    print(f"  Accuracy: {previous_correct}/{previous_total} = {previous_accuracy:.2f}%")
    print(f"\nThree-Agent System:")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.2f}%")
    
    improvement = accuracy - previous_accuracy
    if improvement > 0:
        print(f"\n✅ Improvement: +{improvement:.2f}%")
    elif improvement < 0:
        print(f"\n⚠️  Decrease: {improvement:.2f}%")
    else:
        print(f"\n➡️  No change: {improvement:.2f}%")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()





