"""
Test script for the Three-Agent System (Generator -> Validator -> Reflector)

This script demonstrates how the three-agent system integrates LEAN verification
into MCTS for mathematical reasoning.
"""

import os
import sys
from mcts_parallel import MathReasoningSystemParallel, MathProblem

def test_three_agent_system():
    """Test the three-agent system on a simple math problem"""
    
    print("=" * 70)
    print("THREE-AGENT SYSTEM TEST")
    print("=" * 70)
    print("\nSystem Components:")
    print("  1. LEAN Generator: Creates LEAN proof sketches/subgoals (LLM)")
    print("  2. Proof Validator: Validates subgoals via LEAN server (Symbolic)")
    print("  3. Reflector: Analyzes both outputs to produce numerical signal (LLM + Symbolic)")
    print("\n" + "=" * 70 + "\n")
    
    # Check if LEAN server is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✅ LEAN 4 server is running\n")
        else:
            print("⚠️  LEAN 4 server returned non-200 status\n")
    except Exception as e:
        print(f"⚠️  LEAN 4 server not reachable: {e}")
        print("   Please start the LEAN server with: python real_lean_server.py\n")
    
    # Create a simple test problem
    test_problem = MathProblem(
        problem_text="Find the value of x if 2x + 5 = 15",
        solution="Subtract 5 from both sides: 2x = 10. Divide by 2: x = 5",
        answer="5",
        subject="Algebra",
        level=1,
        has_diagram=False
    )
    
    print(f"Test Problem: {test_problem.problem_text}")
    print(f"Subject: {test_problem.subject}, Level: {test_problem.level}\n")
    
    # Initialize system with three-agent feedback
    print("Initializing Three-Agent System...")
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
    
    # Set max_iterations for testing (modify solver directly)
    system.solver.max_iterations = 10  # Small number for testing
    
    print("\n" + "=" * 70)
    print("SOLVING PROBLEM WITH THREE-AGENT SYSTEM")
    print("=" * 70 + "\n")
    
    # Solve the problem
    result = system.solve_problem(test_problem)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Solution found: {result.get('is_correct', False)}")
    print(f"Predicted answer: {result.get('predicted_answer', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Value: {result.get('value', 0):.2f}")
    print(f"Iterations: {result.get('iterations', 0)}")
    
    # Check for three-agent feedback stats
    feedback_stats = result.get('feedback_stats', {})
    if feedback_stats:
        print("\nThree-Agent Feedback Statistics:")
        for key, value in feedback_stats.items():
            print(f"  {key}: {value}")
    
    # Check feedback type
    feedback_type = result.get('feedback_type', 'None')
    print(f"\nFeedback Type: {feedback_type}")
    
    if feedback_type == "ThreeAgentFeedbackModule":
        print("✅ Three-Agent System successfully integrated!")
    
    print("\n" + "=" * 70)
    print("THREE-AGENT SYSTEM FLOW SUMMARY")
    print("=" * 70)
    print("""
For each MCTS expansion:
  1. Generator (LLM) → Creates LEAN proof sketch/subgoal
  2. Validator (LEAN) → Validates the subgoal via LEAN server
  3. Reflector (LLM + LEAN) → Produces numerical signal [-1.0, 1.0]
  4. MCTS Backpropagation → Uses numerical signal to update value function
  5. Tree Pruning → Reflector recommendation can trigger pruning
    """)
    
    return result


if __name__ == "__main__":
    test_three_agent_system()

