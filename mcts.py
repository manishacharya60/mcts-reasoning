import numpy as np
import json
import time
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from dotenv import load_dotenv
import re
from abc import ABC, abstractmethod

# For LLM integration
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback

# For dataset handling
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class MathProblem:
    """Structured representation of a MATH dataset problem"""
    problem_text: str
    solution: str
    answer: str
    subject: str
    level: int
    has_diagram: bool = False
    diagram_description: str = ""
    
    @classmethod
    def from_dataset(cls, data: Dict[str, Any]) -> 'MathProblem':
        """Create MathProblem from dataset entry"""
        problem_text = data['problem']
        
        # Check for and handle Asymptote diagrams
        has_diagram = '[asy]' in problem_text
        diagram_description = ""
        
        if has_diagram:
            # Extract diagram information
            asy_pattern = r'\[asy\](.*?)\[/asy\]'
            asy_matches = re.findall(asy_pattern, problem_text, re.DOTALL)
            if asy_matches:
                # Parse Asymptote code to understand the diagram
                diagram_description = cls._parse_asymptote(asy_matches[0])
                # Keep the problem text but note that there's a diagram
                problem_text = re.sub(asy_pattern, f"\n[DIAGRAM: {diagram_description}]\n", problem_text, flags=re.DOTALL)
        
        # Extract answer from solution
        answer_match = re.search(r'(?:answer|solution).*?(?:is|=|:)\s*(?:\$|\\boxed{)?([\d\w\s\+\-\*/\^\.]+)(?:\$|\})?', 
                               data['solution'], re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else ""
        
        return cls(
            problem_text=problem_text,
            solution=data['solution'],
            answer=answer,
            subject=data.get('subject', 'Unknown'),
            level=data.get('level', 1),
            has_diagram=has_diagram,
            diagram_description=diagram_description
        )
    
    @staticmethod
    def _parse_asymptote(asy_code: str) -> str:
        """Parse Asymptote code to extract diagram information"""
        description_parts = []
        
        # Extract points
        point_pattern = r'(\w+)\s*=\s*\(([-\d.]+),\s*([-\d.]+)\)'
        points = re.findall(point_pattern, asy_code)
        if points:
            description_parts.append(f"Points: {', '.join([f'{p[0]} at ({p[1]}, {p[2]})' for p in points])}")
        
        # Extract labels
        label_pattern = r'label\("([^"]+)",\s*([^,]+)'
        labels = re.findall(label_pattern, asy_code)
        if labels:
            description_parts.append(f"Labels: {', '.join([f'{l[0]} at {l[1]}' for l in labels])}")
        
        # Extract edge lengths
        edge_pattern = r'label\("(\d+)"'
        edges = re.findall(edge_pattern, asy_code)
        if edges:
            description_parts.append(f"Edge lengths: {', '.join(edges)}")
        
        return "; ".join(description_parts) if description_parts else "Geometric diagram"


@dataclass
class ReasoningState:
    """Represents the current state of mathematical reasoning"""
    problem: MathProblem
    steps: List[str] = field(default_factory=list)
    current_expression: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: List[str] = field(default_factory=list)
    techniques_used: List[str] = field(default_factory=list)
    is_complete: bool = False
    is_correct: bool = False
    solution: Optional[str] = None
    confidence: float = 0.0
    
    def to_string(self) -> str:
        """Convert state to string representation for LLM"""
        state_str = f"Problem ({self.problem.subject} - Level {self.problem.level}):\n"
        state_str += f"{self.problem.problem_text}\n"
        
        if self.problem.has_diagram:
            state_str += f"Note: This problem includes a diagram with {self.problem.diagram_description}\n"
        
        if self.steps:
            state_str += "\nSteps taken:\n"
            for i, step in enumerate(self.steps, 1):
                state_str += f"{i}. {step}\n"
        
        if self.intermediate_results:
            state_str += f"\nIntermediate results: {', '.join(self.intermediate_results)}\n"
        
        if self.current_expression:
            state_str += f"\nCurrent expression: {self.current_expression}\n"
            
        return state_str
    
    def copy(self) -> 'ReasoningState':
        """Create a deep copy of the state"""
        return ReasoningState(
            problem=self.problem,
            steps=self.steps.copy(),
            current_expression=self.current_expression,
            variables=self.variables.copy(),
            intermediate_results=self.intermediate_results.copy(),
            techniques_used=self.techniques_used.copy(),
            is_complete=self.is_complete,
            is_correct=self.is_correct,
            solution=self.solution,
            confidence=self.confidence
        )


@dataclass
class ReasoningAction:
    """Represents a reasoning action/step"""
    description: str
    action_type: str  # Subject-specific action types
    expected_result: str = ""
    confidence: float = 0.5
    
    def __str__(self):
        return f"{self.action_type}: {self.description} (conf: {self.confidence:.2f})"


class SubjectSpecificReasoner:
    """Provides subject-specific reasoning strategies"""
    
    SUBJECT_STRATEGIES = {
        "Algebra": ["solve_equation", "factor", "expand", "substitute", "isolate_variable"],
        "Intermediate Algebra": ["quadratic_formula", "complete_square", "factor", "rational_expressions", "systems_of_equations"],
        "Precalculus": ["trigonometry", "functions", "limits", "sequences", "complex_numbers"],
        "Geometry": ["area_calculation", "perimeter", "similarity", "congruence", "coordinate_geometry", "pythagorean"],
        "Number Theory": ["divisibility", "prime_factorization", "modular_arithmetic", "gcd_lcm", "diophantine"],
        "Counting & Probability": ["combinations", "permutations", "probability", "counting_principle", "expected_value"],
        "Prealgebra": ["arithmetic", "fractions", "percentages", "ratios", "basic_equations"]
    }
    
    @classmethod
    def get_strategies(cls, subject: str) -> List[str]:
        """Get relevant strategies for a subject"""
        return cls.SUBJECT_STRATEGIES.get(subject, ["general_reasoning"])


class LLMReasoner:
    """Handles LLM interactions for mathematical reasoning"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.action_generation_prompt = PromptTemplate(
            input_variables=["state", "strategies", "num_actions"],
            template="""You are an expert mathematical problem solver. Given the current state of a problem, generate possible next reasoning steps.

Current State:
{state}

Relevant strategies for this problem type: {strategies}

Generate {num_actions} different possible next steps. Each step should:
1. Be mathematically valid and precise
2. Make clear progress toward solving the problem
3. Use appropriate techniques for the problem type
4. Consider any diagram or visual information mentioned

Format your response as a JSON list of objects, each with:
- "description": A clear description of the step
- "action_type": The specific technique being used
- "expected_result": What we expect to achieve with this step
- "confidence": A number between 0 and 1 indicating how promising this step is

Your response:"""
        )
        
        self.evaluation_prompt = PromptTemplate(
            input_variables=["state", "true_answer"],
            template="""Evaluate the current state of this mathematical problem solving attempt.

Current State:
{state}

The correct answer to this problem is: {true_answer}

Provide your evaluation as a JSON object with:
- "is_complete": boolean - whether the problem is fully solved
- "is_correct": boolean - whether the solution matches the correct answer
- "progress": number between 0 and 1 - how close to completion
- "quality": number between 0 and 1 - quality of the reasoning so far
- "solution": string or null - the final answer if complete
- "confidence": number between 0 and 1 - confidence in the solution
- "feedback": string - brief explanation of the evaluation

Your response:"""
        )
        
        self.step_application_prompt = PromptTemplate(
            input_variables=["state", "action"],
            template="""Apply the following reasoning step to the current problem state.

Current State:
{state}

Step to apply: {action}

Provide the result as a JSON object with:
- "new_expression": The resulting expression or state after applying the step
- "intermediate_result": Any intermediate result obtained
- "explanation": Brief explanation of what was done
- "success": boolean - whether the step was successfully applied

Your response:"""
        )
    
    def generate_actions(self, state: ReasoningState, num_actions: int = 3) -> List[ReasoningAction]:
        """Generate possible next reasoning steps"""
        strategies = SubjectSpecificReasoner.get_strategies(state.problem.subject)
        
        try:
            with get_openai_callback() as cb:
                response = self.llm.invoke(
                    self.action_generation_prompt.format(
                        state=state.to_string(),
                        strategies=", ".join(strategies),
                        num_actions=num_actions
                    )
                )
                
            # Parse JSON response
            content = response.content
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                actions_data = json.loads(json_match.group())
                actions = []
                for data in actions_data:
                    actions.append(ReasoningAction(
                        description=data.get("description", ""),
                        action_type=data.get("action_type", "unknown"),
                        expected_result=data.get("expected_result", ""),
                        confidence=data.get("confidence", 0.5)
                    ))
                return actions
            else:
                logger.warning("Failed to parse actions from LLM response")
                return []
                
        except Exception as e:
            logger.error(f"Error generating actions: {e}")
            return []
    
    def apply_action(self, state: ReasoningState, action: ReasoningAction) -> ReasoningState:
        """Apply an action to create a new state"""
        try:
            with get_openai_callback() as cb:
                response = self.llm.invoke(
                    self.step_application_prompt.format(
                        state=state.to_string(),
                        action=f"{action.action_type}: {action.description}"
                    )
                )
            
            # Parse response
            content = response.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            new_state = state.copy()
            new_state.steps.append(action.description)
            new_state.techniques_used.append(action.action_type)
            
            if json_match:
                result = json.loads(json_match.group())
                if result.get("success", False):
                    if result.get("new_expression"):
                        new_state.current_expression = result["new_expression"]
                    if result.get("intermediate_result"):
                        new_state.intermediate_results.append(result["intermediate_result"])
            
            return new_state
            
        except Exception as e:
            logger.error(f"Error applying action: {e}")
            # Return state with just the step added
            new_state = state.copy()
            new_state.steps.append(action.description)
            return new_state
    
    def evaluate_state(self, state: ReasoningState) -> Dict[str, Any]:
        """Evaluate the current reasoning state"""
        try:
            with get_openai_callback() as cb:
                response = self.llm.invoke(
                    self.evaluation_prompt.format(
                        state=state.to_string(),
                        true_answer=state.problem.answer
                    )
                )
                
            # Parse JSON response
            content = response.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                return evaluation
            else:
                return self._default_evaluation()
                
        except Exception as e:
            logger.error(f"Error evaluating state: {e}")
            return self._default_evaluation()
    
    def _default_evaluation(self) -> Dict[str, Any]:
        """Return default evaluation when parsing fails"""
        return {
            "is_complete": False,
            "is_correct": False,
            "progress": 0.0,
            "quality": 0.0,
            "solution": None,
            "confidence": 0.0,
            "feedback": "Evaluation failed"
        }


class MCTSNode:
    """Node in the MCTS tree for mathematical reasoning"""
    
    def __init__(self, state: ReasoningState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[ReasoningAction] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[ReasoningAction] = []
        self.is_terminal = False
        self.evaluation_cache: Optional[Dict[str, Any]] = None
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0
    
    def best_child(self, c: float = 1.414) -> 'MCTSNode':
        """Select best child using UCB1"""
        return max(self.children, key=lambda child: child.ucb1_value(c))
    
    def ucb1_value(self, c: float) -> float:
        """Calculate UCB1 value for this node"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = c * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def add_child(self, state: ReasoningState, action: ReasoningAction) -> 'MCTSNode':
        """Add a child node"""
        child = MCTSNode(state, parent=self, action=action)
        self.children.append(child)
        return child
    
    def update(self, value: float):
        """Update node statistics"""
        self.visits += 1
        self.value += value
    
    def get_path(self) -> List[Tuple[Optional[ReasoningAction], ReasoningState]]:
        """Get the path from root to this node"""
        path = []
        node = self
        while node is not None:
            path.append((node.action, node.state))
            node = node.parent
        return list(reversed(path))


class MCTSReasoningSolver:
    """MCTS-based mathematical reasoning solver"""
    
    def __init__(self, llm_reasoner: LLMReasoner, max_iterations: int = 100, 
                 exploration_constant: float = 1.414, max_depth: int = 20):
        self.llm_reasoner = llm_reasoner
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.stats = defaultdict(int)
    
    def solve(self, problem: MathProblem) -> Dict[str, Any]:
        """Solve a mathematical problem using MCTS + LLM"""
        # Initialize root state
        initial_state = ReasoningState(problem=problem)
        root = MCTSNode(initial_state)
        
        best_solution = None
        best_value = -float('inf')
        
        logger.info(f"Starting MCTS for {problem.subject} problem (Level {problem.level})")
        
        for iteration in range(self.max_iterations):
            # MCTS phases
            node = self._select(root)
            
            if not node.is_terminal and len(node.get_path()) < self.max_depth:
                node = self._expand(node)
            
            value = self._simulate(node)
            self._backpropagate(node, value)
            
            # Track best solution
            if node.evaluation_cache and node.evaluation_cache.get("is_complete"):
                solution_value = value * node.evaluation_cache.get("confidence", 1.0)
                if solution_value > best_value:
                    best_value = solution_value
                    best_solution = {
                        "solution": node.evaluation_cache.get("solution"),
                        "is_correct": node.evaluation_cache.get("is_correct", False),
                        "reasoning_path": [(a.description if a else "Initial state", s) 
                                         for a, s in node.get_path()],
                        "value": solution_value,
                        "confidence": node.evaluation_cache.get("confidence", 0),
                        "iterations": iteration + 1,
                        "techniques_used": node.state.techniques_used
                    }
            
            # Early stopping if we found a correct solution with high confidence
            if best_solution and best_solution.get("is_correct") and best_value > 0.95:
                logger.info(f"Found correct solution after {iteration + 1} iterations")
                break
        
        # If no complete solution found, return the best partial solution
        if not best_solution:
            best_node = max(root.children, key=lambda n: n.value / n.visits if n.visits > 0 else 0)
            evaluation = self.llm_reasoner.evaluate_state(best_node.state)
            best_solution = {
                "solution": evaluation.get("solution", "No complete solution found"),
                "is_correct": False,
                "reasoning_path": [(a.description if a else "Initial state", s) 
                                 for a, s in best_node.get_path()],
                "value": best_node.value / best_node.visits if best_node.visits > 0 else 0,
                "confidence": evaluation.get("confidence", 0),
                "iterations": self.max_iterations,
                "techniques_used": best_node.state.techniques_used
            }
        
        best_solution["stats"] = dict(self.stats)
        return best_solution
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCB1"""
        while node.children and node.is_fully_expanded() and not node.is_terminal:
            node = node.best_child(self.exploration_constant)
            self.stats['selections'] += 1
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase: add new child node"""
        if not node.untried_actions:
            # Generate new actions using LLM
            actions = self.llm_reasoner.generate_actions(node.state, num_actions=3)
            # Sort by confidence to try most promising first
            node.untried_actions = sorted(actions, key=lambda a: a.confidence, reverse=True)
        
        if node.untried_actions:
            action = node.untried_actions.pop(0)
            
            # Apply action to create new state
            new_state = self.llm_reasoner.apply_action(node.state, action)
            child = node.add_child(new_state, action)
            
            # Evaluate if terminal
            evaluation = self.llm_reasoner.evaluate_state(new_state)
            child.evaluation_cache = evaluation
            child.is_terminal = evaluation.get("is_complete", False)
            
            self.stats['expansions'] += 1
            return child
        
        return node
    
    def _simulate(self, node: MCTSNode) -> float:
        """Simulation phase: estimate value through evaluation"""
        if node.evaluation_cache:
            evaluation = node.evaluation_cache
        else:
            evaluation = self.llm_reasoner.evaluate_state(node.state)
            node.evaluation_cache = evaluation
        
        # Calculate value based on multiple factors
        if evaluation.get("is_complete") and evaluation.get("is_correct"):
            value = 1.0
        elif evaluation.get("is_complete"):
            value = 0.2  # Completed but incorrect
        else:
            # Partial credit based on progress and quality
            progress = evaluation.get("progress", 0.0)
            quality = evaluation.get("quality", 0.0)
            confidence = evaluation.get("confidence", 0.5)
            value = 0.4 * progress + 0.4 * quality + 0.2 * confidence
        
        self.stats['simulations'] += 1
        return value
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagation phase: update statistics"""
        while node is not None:
            node.update(value)
            node = node.parent
            self.stats['backpropagations'] += 1


class MathReasoningSystem:
    """Complete system for mathematical reasoning with MCTS + LLM"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        self.llm_reasoner = LLMReasoner(model_name=model_name, temperature=temperature)
        self.solver = MCTSReasoningSolver(
            llm_reasoner=self.llm_reasoner,
            max_iterations=50,
            exploration_constant=1.414,
            max_depth=15
        )
    
    def solve_problem(self, problem: Union[str, MathProblem, Dict[str, Any]]) -> Dict[str, Any]:
        """Solve a single mathematical problem"""
        start_time = time.time()
        
        # Convert input to MathProblem
        if isinstance(problem, str):
            math_problem = MathProblem(
                problem_text=problem,
                solution="",
                answer="",
                subject="Unknown",
                level=1
            )
        elif isinstance(problem, dict):
            math_problem = MathProblem.from_dataset(problem)
        else:
            math_problem = problem
        
        try:
            result = self.solver.solve(math_problem)
            result['time_taken'] = time.time() - start_time
            result['problem_info'] = {
                'subject': math_problem.subject,
                'level': math_problem.level,
                'has_diagram': math_problem.has_diagram
            }
            return result
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return {
                "solution": None,
                "error": str(e),
                "time_taken": time.time() - start_time
            }
    
    def evaluate_on_dataset(self, dataset_name: str = "HuggingFaceH4/MATH-500", 
                          split: str = "test", 
                          num_problems: int = 10,
                          subjects: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate on MATH dataset"""
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        
        results = []
        results_by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
        
        # Filter by subjects if specified
        if subjects:
            indices = [i for i, item in enumerate(dataset) 
                      if item.get('subject') in subjects]
        else:
            indices = list(range(len(dataset)))
        
        # Sample problems
        sample_indices = np.random.choice(indices, 
                                        min(num_problems, len(indices)), 
                                        replace=False)
        
        for idx in sample_indices:
            problem_data = dataset[int(idx)]
            
            logger.info(f"\nSolving problem {len(results) + 1}/{num_problems}")
            logger.info(f"Subject: {problem_data.get('subject', 'Unknown')}")
            logger.info(f"Level: {problem_data.get('level', 'Unknown')}")
            
            result = self.solve_problem(problem_data)
            
            # Check correctness
            is_correct = result.get("is_correct", False)
            
            # Update statistics
            subject = problem_data.get('subject', 'Unknown')
            results_by_subject[subject]["total"] += 1
            if is_correct:
                results_by_subject[subject]["correct"] += 1
            
            result['true_answer'] = problem_data.get('answer', 
                                                    problem_data.get('solution', ''))
            results.append(result)
            
            # Log progress
            logger.info(f"Predicted: {result.get('solution', 'No solution')}")
            logger.info(f"Correct: {is_correct}")
            total_correct = sum(r["correct"] for r in results_by_subject.values())
            total_problems = len(results)
            logger.info(f"Overall accuracy: {total_correct}/{total_problems} = "
                       f"{total_correct/total_problems:.2%}")
        
        # Calculate overall statistics
        total_correct = sum(r["correct"] for r in results_by_subject.values())
        total_problems = sum(r["total"] for r in results_by_subject.values())
        
        return {
            "overall_accuracy": total_correct / total_problems if total_problems > 0 else 0,
            "problems_solved": total_problems,
            "correct": total_correct,
            "results_by_subject": dict(results_by_subject),
            "results": results,
            "average_time": np.mean([r.get('time_taken', 0) for r in results]),
            "average_iterations": np.mean([r.get('iterations', 0) for r in results])
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    system = MathReasoningSystem(model_name="gpt-4o-mini", temperature=0.7)
    
    # Example 1: Simple problem without dataset structure
    print("=== Example 1: Simple Algebra ===")
    problem1 = "Solve for x: 2x + 5 = 13"
    result1 = system.solve_problem(problem1)
    print(f"Solution: {result1.get('solution')}")
    print(f"Correct: {result1.get('is_correct')}")
    print(f"Confidence: {result1.get('confidence', 0):.2%}")
    
    # Example 2: Problem with dataset structure
    print("\n=== Example 2: Geometry Problem ===")
    problem2_data = {
        'problem': "Find the area of a triangle with vertices at (0,0), (3,0), and (0,4).",
        'solution': "Using the formula for the area of a triangle with vertices at coordinates, "
                   "we can use the base and height. The base is 3 and the height is 4. "
                   "Area = (1/2) × base × height = (1/2) × 3 × 4 = 6",
        'answer': "6",
        'subject': "Geometry",
        'level': 2
    }
    result2 = system.solve_problem(problem2_data)
    print(f"Solution: {result2.get('solution')}")
    print(f"Correct: {result2.get('is_correct')}")
    print(f"Techniques used: {result2.get('techniques_used', [])}")
    
    # Example 3: Evaluate on specific subjects
    # print("\n=== Dataset Evaluation ===")
    # eval_results = system.evaluate_on_dataset(
    #     num_problems=5,
    #     subjects=["Algebra", "Geometry"]
    # )
    # print(f"Overall Accuracy: {eval_results['overall_accuracy']:.2%}")
    # print("\nResults by Subject:")
    # for subject, stats in eval_results['results_by_subject'].items():
    #     if stats['total'] > 0:
    #         acc = stats['correct'] / stats['total']
    #         print(f"  {subject}: {stats['correct']}/{stats['total']} = {acc:.2%}")