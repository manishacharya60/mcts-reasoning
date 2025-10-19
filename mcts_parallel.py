"""
Parallel Monte Carlo Tree Search for Mathematical Reasoning with Feedback Integration

This implementation extends the original MCTS system with feedback-aware reasoning that incorporates:
- Neural feedback: Numeric feedback based on LLM evaluation of reasoning progress
- Symbolic feedback: LEAN-based formal verification feedback for mathematical proofs
- Combined feedback: Weighted integration of both neural and symbolic feedback sources

The feedback system adjusts MCTS exploration, node expansion, and value updates based on structured
feedback without requiring neural network policy/value models.

Key Features:
- FeedbackInterface: Abstract interface for different feedback types
- NeuralFeedbackModule: LLM-based evaluation feedback
- SymbolicFeedbackModule: LEAN server integration for formal verification
- CombinedFeedbackModule: Weighted combination of feedback sources
- Feedback-aware MCTS nodes with reward tracking
- Thread-safe parallel feedback integration
- Comprehensive logging and visualization of feedback effects
"""

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
import datetime
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# For LLM integration
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback

# For dataset handling
from datasets import load_dataset

# For colored terminal output
import colorama
from colorama import Fore, Back, Style
colorama.init()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


# Manish: added feedback-aware reasoning system
class FeedbackInterface(ABC):
    """Abstract interface for neural and symbolic feedback in MCTS reasoning"""
    
    @abstractmethod
    def get_feedback(self, state: 'ReasoningState', action: 'ReasoningAction') -> Tuple['ReasoningState', float, Dict[str, Any]]:
        """
        Return feedback for a state-action pair.
        
        Args:
            state: Current reasoning state
            action: Action taken
            
        Returns:
            Tuple of (next_state, reward, info)
            - next_state: Updated state after action
            - reward: Numeric feedback in [-1, 1]
            - info: Additional feedback information
        """
        raise NotImplementedError


class NeuralFeedbackModule(FeedbackInterface):
    """Neural feedback module providing numeric feedback based on LLM evaluation"""
    
    def __init__(self, llm_reasoner: 'ParallelLLMReasoner', logger: Optional['DebugLogger'] = None):
        self.llm_reasoner = llm_reasoner
        self.logger = logger or DebugLogger()
        
    def get_feedback(self, state: 'ReasoningState', action: 'ReasoningAction') -> Tuple['ReasoningState', float, Dict[str, Any]]:
        """Get neural feedback by applying action and evaluating result"""
        try:
            # Apply action to get next state
            next_state = self.llm_reasoner.apply_action(state, action)
            
            # Evaluate the new state
            evaluation = self.llm_reasoner.evaluate_state(next_state)
            
            # Calculate reward based on evaluation
            if evaluation.get("is_complete") and evaluation.get("is_correct"):
                reward = 1.0
            elif evaluation.get("is_complete"):
                reward = 0.2  # Partial credit for completion
            else:
                # Combine progress, quality, and confidence
                progress = evaluation.get("progress", 0.0)
                quality = evaluation.get("quality", 0.0)
                confidence = evaluation.get("confidence", 0.5)
                reward = 0.4 * progress + 0.4 * quality + 0.2 * confidence
                reward = max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]
            
            info = {
                "feedback_type": "neural",
                "evaluation": evaluation,
                "action_applied": action.description,
                "technique_used": action.action_type
            }
            
            self.logger.log(f"Neural feedback: reward={reward:.3f}, progress={evaluation.get('progress', 0):.3f}", 
                          Fore.BLUE)
            
            return next_state, reward, info
            
        except Exception as e:
            self.logger.log(f"Error in neural feedback: {e}", Fore.RED)
            # Return neutral feedback on error
            return state, 0.0, {"feedback_type": "neural", "error": str(e)}


class SymbolicFeedbackModule(FeedbackInterface):
    """Symbolic feedback module for LEAN-based formal verification"""
    
    def __init__(self, lean_server_url: str = "http://localhost:8000", 
                 logger: Optional['DebugLogger'] = None):
        self.lean_server_url = lean_server_url
        self.logger = logger or DebugLogger()
        self._session = None
        
    def _get_session(self):
        """Get or create HTTP session for LEAN server communication"""
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session
    
    def get_feedback(self, state: 'ReasoningState', action: 'ReasoningAction') -> Tuple['ReasoningState', float, Dict[str, Any]]:
        """Get symbolic feedback from LEAN server"""
        try:
            # Convert reasoning state to LEAN proof state
            proof_state = self._state_to_lean_proof(state)
            
            # Convert action to LEAN tactic
            tactic = self._action_to_lean_tactic(action)
            
            # Send to LEAN server
            response = self._query_lean_server(proof_state, tactic)
            
            if response.get("success", False):
                # Success: tactic worked
                next_state = state.copy()
                next_state.steps.append(action.description)
                next_state.techniques_used.append(action.action_type)
                
                # Small positive reward for successful tactic
                reward = 0.1
                
                info = {
                    "feedback_type": "symbolic",
                    "lean_response": response,
                    "tactic_success": True,
                    "proof_progress": response.get("proof_progress", 0.0)
                }
                
                self.logger.log(f"Symbolic feedback: tactic succeeded, reward={reward:.3f}", 
                              Fore.GREEN)
                
            else:
                # Failure: tactic didn't work
                next_state = state.copy()
                next_state.steps.append(f"Failed: {action.description}")
                
                # Negative reward for failed tactic
                reward = -0.5
                
                info = {
                    "feedback_type": "symbolic",
                    "lean_response": response,
                    "tactic_success": False,
                    "error_message": response.get("error", "Unknown error")
                }
                
                self.logger.log(f"Symbolic feedback: tactic failed, reward={reward:.3f}", 
                              Fore.RED)
            
            return next_state, reward, info
            
        except Exception as e:
            self.logger.log(f"Error in symbolic feedback: {e}", Fore.RED)
            # Return neutral feedback on error
            return state, 0.0, {"feedback_type": "symbolic", "error": str(e)}
    
    def _state_to_lean_proof(self, state: 'ReasoningState') -> str:
        """Convert reasoning state to LEAN proof state representation"""
        # This is a simplified conversion - in practice, you'd need more sophisticated
        # translation from mathematical reasoning to formal proof states
        proof_parts = []
        
        if state.problem.problem_text:
            proof_parts.append(f"-- Problem: {state.problem.problem_text[:100]}...")
        
        if state.steps:
            proof_parts.append("-- Steps taken:")
            for i, step in enumerate(state.steps):
                proof_parts.append(f"-- {i+1}. {step}")
        
        if state.current_expression:
            proof_parts.append(f"-- Current: {state.current_expression}")
        
        return "\n".join(proof_parts)
    
    def _action_to_lean_tactic(self, action: 'ReasoningAction') -> str:
        """Convert reasoning action to LEAN tactic"""
        # Map action types to LEAN tactics
        tactic_mapping = {
            "solve_equation": "simp",
            "factor": "ring",
            "expand": "ring",
            "substitute": "rw",
            "isolate_variable": "simp",
            "quadratic_formula": "ring",
            "complete_square": "ring",
            "trigonometry": "simp",
            "functions": "simp",
            "geometry": "simp",
            "number_theory": "simp"
        }
        
        # Get base tactic from action type
        base_tactic = tactic_mapping.get(action.action_type, "simp")
        
        # Add description as comment
        return f"-- {action.description}\n{base_tactic}"
    
    def _query_lean_server(self, proof_state: str, tactic: str) -> Dict[str, Any]:
        """Query LEAN server with proof state and tactic"""
        try:
            import requests
            
            payload = {
                "proof_state": proof_state,
                "tactic": tactic
            }
            
            session = self._get_session()
            response = session.post(f"{self.lean_server_url}/verify", 
                                  json=payload, 
                                  timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "error": f"Server error: {response.status_code}",
                    "proof_progress": 0.0
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Connection error: {str(e)}",
                "proof_progress": 0.0
            }
    
    def __del__(self):
        """Cleanup session"""
        if self._session:
            self._session.close()


class CombinedFeedbackModule(FeedbackInterface):
    """Combined feedback module that integrates both neural and symbolic feedback"""
    
    def __init__(self, neural_feedback: NeuralFeedbackModule, 
                 symbolic_feedback: SymbolicFeedbackModule,
                 neural_weight: float = 0.7, symbolic_weight: float = 0.3,
                 logger: Optional['DebugLogger'] = None):
        self.neural_feedback = neural_feedback
        self.symbolic_feedback = symbolic_feedback
        self.neural_weight = neural_weight
        self.symbolic_weight = symbolic_weight
        self.logger = logger or DebugLogger()
        
    def get_feedback(self, state: 'ReasoningState', action: 'ReasoningAction') -> Tuple['ReasoningState', float, Dict[str, Any]]:
        """Get combined feedback from both neural and symbolic sources"""
        try:
            # Get neural feedback
            neural_state, neural_reward, neural_info = self.neural_feedback.get_feedback(state, action)
            
            # Get symbolic feedback
            symbolic_state, symbolic_reward, symbolic_info = self.symbolic_feedback.get_feedback(state, action)
            
            # Combine rewards with weights
            total_reward = (self.neural_weight * neural_reward + 
                          self.symbolic_weight * symbolic_reward)
            
            # Use neural state as primary (it has more detailed reasoning)
            combined_state = neural_state
            
            # Combine info
            combined_info = {
                "feedback_type": "combined",
                "neural_feedback": neural_info,
                "symbolic_feedback": symbolic_info,
                "neural_reward": neural_reward,
                "symbolic_reward": symbolic_reward,
                "combined_reward": total_reward,
                "weights": {
                    "neural": self.neural_weight,
                    "symbolic": self.symbolic_weight
                }
            }
            
            self.logger.log(f"Combined feedback: neural={neural_reward:.3f}, symbolic={symbolic_reward:.3f}, "
                          f"total={total_reward:.3f}", Fore.MAGENTA)
            
            return combined_state, total_reward, combined_info
            
        except Exception as e:
            self.logger.log(f"Error in combined feedback: {e}", Fore.RED)
            # Fallback to neural feedback only
            return self.neural_feedback.get_feedback(state, action)


class DebugLogger:
    """Enhanced logging for MCTS reasoning process"""
    
    def __init__(self, verbose: bool = True, log_llm_calls: bool = True):
        self.verbose = verbose
        self.log_llm_calls = log_llm_calls
        self.indent_level = 0
        self.iteration_count = 0
        self._lock = threading.Lock()  # Thread safety for logging
        
    def indent(self):
        self.indent_level += 1
        
    def dedent(self):
        self.indent_level = max(0, self.indent_level - 1)
        
    def log(self, message: str, color: str = Fore.WHITE, important: bool = False):
        if self.verbose:
            with self._lock:
                indent = "  " * self.indent_level
                if important:
                    print(f"{color}{Style.BRIGHT}{indent}{message}{Style.RESET_ALL}")
                else:
                    print(f"{color}{indent}{message}{Style.RESET_ALL}")
    
    def log_iteration(self, iteration: int, max_iterations: int):
        self.iteration_count = iteration
        self.log(f"\n{'='*60}", Fore.CYAN, important=True)
        self.log(f"MCTS ITERATION {iteration}/{max_iterations}", Fore.CYAN, important=True)
        self.log(f"{'='*60}", Fore.CYAN, important=True)
    
    def log_phase(self, phase: str):
        self.log(f"\n[{phase.upper()}]", Fore.YELLOW, important=True)
    
    def log_action(self, action: 'ReasoningAction', selected: bool = False):
        if selected:
            self.log(f"✓ SELECTED: {action}", Fore.GREEN, important=True)
        else:
            self.log(f"  • {action}", Fore.WHITE)
    
    def log_llm_call(self, prompt_type: str, input_data: str, response: str):
        if self.log_llm_calls:
            self.log(f"\n{'─'*40}", Fore.MAGENTA)
            self.log(f"LLM CALL: {prompt_type}", Fore.MAGENTA, important=True)
            self.log("Input:", Fore.MAGENTA)
            self.indent()
            # Truncate long inputs
            if len(input_data) > 500:
                self.log(input_data[:500] + "...[truncated]", Fore.WHITE)
            else:
                self.log(input_data, Fore.WHITE)
            self.dedent()
            self.log("Response:", Fore.MAGENTA)
            self.indent()
            if len(response) > 500:
                self.log(response[:500] + "...[truncated]", Fore.WHITE)
            else:
                self.log(response, Fore.WHITE)
            self.dedent()
            self.log(f"{'─'*40}", Fore.MAGENTA)
    
    def log_problem_start(self, problem_num: int, total: int, subject: str, level: int):
        self.log(f"\n{'━'*80}", Fore.CYAN, important=True)
        self.log(f"PROBLEM {problem_num}/{total} - {subject} (Level {level})", 
                Fore.CYAN, important=True)
        self.log(f"{'━'*80}", Fore.CYAN, important=True)
    
    def log_reasoning_step(self, step_num: int, action: str, result: str = None):
        self.log(f"\n📝 Step {step_num}: {action}", Fore.BLUE, important=True)
        if result:
            self.indent()
            self.log(f"Result: {result}", Fore.GREEN)
            self.dedent()


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
            # Extract diagram information without regex
            start = problem_text.find('[asy]')
            end = problem_text.find('[/asy]')
            if start != -1 and end != -1:
                asy_code = problem_text[start+5:end]
                diagram_description = cls._parse_asymptote(asy_code)
                # Replace diagram with description
                problem_text = problem_text[:start] + f"\n[DIAGRAM: {diagram_description}]\n" + problem_text[end+6:]
        
        # Extract answer from solution WITHOUT REGEX to avoid escape errors
        answer = ""
        solution_text = data.get('solution', '')
        
        # Method 1: Look for \boxed{...} without regex
        if '\\boxed{' in solution_text:
            start = solution_text.find('\\boxed{')
            if start != -1:
                start += 7  # length of '\boxed{'
                # Find matching closing brace
                brace_count = 1
                end = start
                while end < len(solution_text) and brace_count > 0:
                    if solution_text[end] == '{':
                        brace_count += 1
                    elif solution_text[end] == '}':
                        brace_count -= 1
                    end += 1
                if brace_count == 0:
                    answer = solution_text[start:end-1].strip()
        
        # Method 2: If no boxed answer, look for common patterns
        if not answer:
            # Look for "answer is" or "equals" patterns
            lower_solution = solution_text.lower()
            for pattern in ['answer is', 'answer:', 'equals', '=']:
                idx = lower_solution.rfind(pattern)  # Look for last occurrence
                if idx != -1:
                    # Extract the rest of the line
                    start = idx + len(pattern)
                    end = solution_text.find('\n', start)
                    if end == -1:
                        end = len(solution_text)
                    potential_answer = solution_text[start:end].strip()
                    # Clean up common formatting
                    potential_answer = potential_answer.strip('$. ')
                    if potential_answer and len(potential_answer) < 50:  # Reasonable answer length
                        answer = potential_answer
                        break
        
        # Method 3: Check if there's an 'answer' field in the data
        if not answer and 'answer' in data:
            answer = str(data['answer']).strip()
        
        return cls(
            problem_text=problem_text,
            solution=solution_text,
            answer=answer,
            subject=data.get('subject', 'Unknown'),
            level=data.get('level', 1),
            has_diagram=has_diagram,
            diagram_description=diagram_description
        )
    
    @staticmethod
    def _parse_asymptote(asy_code: str) -> str:
        """Parse Asymptote code to extract diagram information without regex"""
        description_parts = []
        
        # Extract points by looking for pattern: letter = (number, number)
        lines = asy_code.split('\n')
        points = []
        labels = []
        
        for line in lines:
            # Look for point definitions
            if '=' in line and '(' in line and ')' in line:
                eq_idx = line.find('=')
                paren_start = line.find('(', eq_idx)
                paren_end = line.find(')', paren_start)
                
                if eq_idx > 0 and paren_start > eq_idx and paren_end > paren_start:
                    # Extract point name
                    point_name = line[:eq_idx].strip().split()[-1] if line[:eq_idx].strip() else ""
                    # Extract coordinates
                    coords = line[paren_start+1:paren_end]
                    if point_name and ',' in coords:
                        points.append(f"{point_name} at ({coords})")
            
            # Look for labels
            if 'label(' in line:
                start = line.find('label(')
                if start != -1:
                    # Find the content between quotes
                    quote1 = line.find('"', start)
                    if quote1 != -1:
                        quote2 = line.find('"', quote1 + 1)
                        if quote2 != -1:
                            label_text = line[quote1+1:quote2]
                            if label_text.isdigit():
                                # It's an edge length
                                labels.append(f"Edge length: {label_text}")
                            else:
                                labels.append(f"Label: {label_text}")
        
        if points:
            description_parts.append(f"Points: {', '.join(points)}")
        if labels:
            description_parts.append(f"{'; '.join(labels)}")
        
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
            # Convert all intermediate results to strings
            intermediate_results_str = [str(r) for r in self.intermediate_results]
            state_str += f"\nIntermediate results: {', '.join(intermediate_results_str)}\n"
        
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


class ParallelLLMReasoner:
    """Handles LLM interactions with parallel execution support"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7, 
                 logger: Optional[DebugLogger] = None, max_workers: int = 3):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.logger = logger or DebugLogger()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
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

IMPORTANT: Format your response as valid JSON. When including mathematical notation:
- Use double backslashes (\\\\) for LaTeX commands (e.g., \\\\sin, \\\\theta, \\\\circ)
- Or better yet, use plain text descriptions when possible

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
    
    def _llm_call(self, prompt: str) -> str:
        """Make an LLM call (can be used in parallel)"""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.log(f"LLM call error: {e}", Fore.RED)
            return ""
    
    def generate_actions_parallel(self, states: List[ReasoningState], num_actions: int = 3) -> List[List[ReasoningAction]]:
        """Generate actions for multiple states in parallel"""
        prompts = []
        for state in states:
            strategies = SubjectSpecificReasoner.get_strategies(state.problem.subject)
            prompt = self.action_generation_prompt.format(
                state=state.to_string(),
                strategies=", ".join(strategies),
                num_actions=num_actions
            )
            prompts.append(prompt)
        
        # Execute LLM calls in parallel
        futures = [self.executor.submit(self._llm_call, prompt) for prompt in prompts]
        results = []
        
        for i, future in enumerate(as_completed(futures)):
            try:
                content = future.result()
                actions = self._parse_actions(content)
                results.append((i, actions))
            except Exception as e:
                self.logger.log(f"Error in parallel action generation: {e}", Fore.RED)
                results.append((i, []))
        
        # Sort results by original order
        results.sort(key=lambda x: x[0])
        return [actions for _, actions in results]
    
    def generate_actions(self, state: ReasoningState, num_actions: int = 3) -> List[ReasoningAction]:
        """Generate possible next reasoning steps"""
        strategies = SubjectSpecificReasoner.get_strategies(state.problem.subject)
        
        prompt_input = self.action_generation_prompt.format(
            state=state.to_string(),
            strategies=", ".join(strategies),
            num_actions=num_actions
        )
        
        try:
            with get_openai_callback() as cb:
                response = self.llm.invoke(prompt_input)
                
            self.logger.log_llm_call("Action Generation", prompt_input, response.content)
            
            return self._parse_actions(response.content)
                
        except Exception as e:
            self.logger.log(f"Error generating actions: {e}", Fore.RED)
            return []
    
    def _parse_actions(self, content: str) -> List[ReasoningAction]:
        """Parse actions from LLM response"""
        try:
            # Find JSON array in the content without using regex on the full content
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx+1]
                
                # Try to parse as-is first
                try:
                    actions_data = json.loads(json_str)
                    actions = []
                    for data in actions_data:
                        actions.append(ReasoningAction(
                            description=data.get("description", ""),
                            action_type=data.get("action_type", "unknown"),
                            expected_result=data.get("expected_result", ""),
                            confidence=data.get("confidence", 0.5)
                        ))
                    return actions
                except json.JSONDecodeError:
                    # If that fails, try with escaped backslashes
                    try:
                        # Simple replacement approach
                        json_str_escaped = json_str.replace('\\', '\\\\')
                        actions_data = json.loads(json_str_escaped)
                        actions = []
                        for data in actions_data:
                            actions.append(ReasoningAction(
                                description=data.get("description", "").replace('\\\\', '\\'),
                                action_type=data.get("action_type", "unknown"),
                                expected_result=data.get("expected_result", ""),
                                confidence=data.get("confidence", 0.5)
                            ))
                        return actions
                    except:
                        pass
            
            # If all else fails, use fallback
            return self._extract_actions_fallback(content)
                
        except Exception as e:
            self.logger.log(f"Error in _parse_actions: {e}", Fore.RED)
            return self._extract_actions_fallback(content)
    
    def _extract_actions_fallback(self, content: str) -> List[ReasoningAction]:
        """Fallback method to extract actions when JSON parsing fails"""
        actions = []
        
        try:
            # Split content into potential JSON objects
            lines = content.split('\n')
            current_obj = []
            in_object = False
            
            for line in lines:
                if '{' in line:
                    in_object = True
                    current_obj = [line]
                elif '}' in line and in_object:
                    current_obj.append(line)
                    obj_str = '\n'.join(current_obj)
                    
                    # Try to extract fields manually
                    description = ""
                    action_type = "unknown"
                    confidence = 0.5
                    
                    # Extract description
                    if '"description"' in obj_str:
                        start = obj_str.find('"description"') + len('"description"')
                        start = obj_str.find('"', start) + 1
                        end = obj_str.find('"', start)
                        if end > start:
                            description = obj_str[start:end]
                    
                    # Extract action_type
                    if '"action_type"' in obj_str:
                        start = obj_str.find('"action_type"') + len('"action_type"')
                        start = obj_str.find('"', start) + 1
                        end = obj_str.find('"', start)
                        if end > start:
                            action_type = obj_str[start:end]
                    
                    # Extract confidence
                    if '"confidence"' in obj_str:
                        start = obj_str.find('"confidence"') + len('"confidence"')
                        start = obj_str.find(':', start) + 1
                        # Find next comma or closing brace
                        end = min((obj_str.find(',', start) if obj_str.find(',', start) != -1 else len(obj_str),
                                  obj_str.find('}', start) if obj_str.find('}', start) != -1 else len(obj_str)))
                        if end > start:
                            try:
                                confidence = float(obj_str[start:end].strip())
                            except:
                                confidence = 0.5
                    
                    if description:
                        actions.append(ReasoningAction(
                            description=description,
                            action_type=action_type,
                            expected_result="",
                            confidence=confidence
                        ))
                    
                    in_object = False
                    current_obj = []
                elif in_object:
                    current_obj.append(line)
            
        except Exception as e:
            self.logger.log(f"Error in fallback extraction: {e}", Fore.RED)
        
        if not actions:
            # If still no actions, create a default one
            actions.append(ReasoningAction(
                description="Attempt to simplify the expression using algebraic manipulation",
                action_type="simplification",
                expected_result="Simplified form",
                confidence=0.5
            ))
        
        return actions[:3]  # Return at most 3 actions
    
    def apply_action(self, state: ReasoningState, action: ReasoningAction) -> ReasoningState:
        """Apply an action to create a new state"""
        prompt_input = self.step_application_prompt.format(
            state=state.to_string(),
            action=f"{action.action_type}: {action.description}"
        )
        
        try:
            with get_openai_callback() as cb:
                response = self.llm.invoke(prompt_input)
            
            self.logger.log_llm_call("Action Application", prompt_input, response.content)
            
            # Parse response
            content = response.content
            
            new_state = state.copy()
            new_state.steps.append(action.description)
            new_state.techniques_used.append(action.action_type)
            
            try:
                # Find JSON object without using regex
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx+1]
                    
                    # Try parsing
                    result = None
                    try:
                        result = json.loads(json_str)
                    except:
                        # Try with escaped backslashes
                        try:
                            json_str_escaped = json_str.replace('\\', '\\\\')
                            result = json.loads(json_str_escaped)
                        except:
                            pass
                    
                    if result and result.get("success", False):
                        if result.get("new_expression"):
                            new_state.current_expression = result["new_expression"]
                        if result.get("intermediate_result"):
                            # Ensure intermediate result is string
                            intermediate_result = str(result["intermediate_result"])
                            new_state.intermediate_results.append(intermediate_result)
                        
                        # Log the reasoning step
                        self.logger.log_reasoning_step(
                            len(new_state.steps), 
                            action.description, 
                            result.get("explanation", "")
                        )
            except Exception as e:
                self.logger.log(f"Error parsing action result: {e}", Fore.YELLOW)
            
            return new_state
            
        except Exception as e:
            self.logger.log(f"Error applying action: {e}", Fore.RED)
            # Return state with just the step added
            new_state = state.copy()
            new_state.steps.append(action.description)
            return new_state
    
    def evaluate_states_parallel(self, states: List[ReasoningState]) -> List[Dict[str, Any]]:
        """Evaluate multiple states in parallel"""
        prompts = []
        for state in states:
            prompt = self.evaluation_prompt.format(
                state=state.to_string(),
                true_answer=state.problem.answer
            )
            prompts.append(prompt)
        
        # Execute LLM calls in parallel
        futures = [self.executor.submit(self._llm_call, prompt) for prompt in prompts]
        results = []
        
        for i, future in enumerate(as_completed(futures)):
            try:
                content = future.result()
                evaluation = self._parse_evaluation(content)
                results.append((i, evaluation))
            except Exception as e:
                self.logger.log(f"Error in parallel evaluation: {e}", Fore.RED)
                results.append((i, self._default_evaluation()))
        
        # Sort results by original order
        results.sort(key=lambda x: x[0])
        return [evaluation for _, evaluation in results]
    
    def evaluate_state(self, state: ReasoningState) -> Dict[str, Any]:
        """Evaluate the current reasoning state"""
        prompt_input = self.evaluation_prompt.format(
            state=state.to_string(),
            true_answer=state.problem.answer
        )
        
        try:
            with get_openai_callback() as cb:
                response = self.llm.invoke(prompt_input)
                
            self.logger.log_llm_call("State Evaluation", prompt_input, response.content)
            
            return self._parse_evaluation(response.content)
                
        except Exception as e:
            self.logger.log(f"Error evaluating state: {e}", Fore.RED)
            return self._default_evaluation()
    
    def _parse_evaluation(self, content: str) -> Dict[str, Any]:
        """Parse evaluation from LLM response"""
        try:
            # Find JSON object in the content without using regex
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx+1]
                
                # Try to parse as-is first
                try:
                    evaluation = json.loads(json_str)
                    return evaluation
                except json.JSONDecodeError:
                    # If that fails, try with escaped backslashes
                    try:
                        json_str_escaped = json_str.replace('\\', '\\\\')
                        evaluation = json.loads(json_str_escaped)
                        return evaluation
                    except:
                        pass
            
            return self._default_evaluation()
            
        except Exception as e:
            self.logger.log(f"Error in _parse_evaluation: {e}", Fore.RED)
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
    
    def shutdown(self):
        """Shutdown the thread pool executor"""
        self.executor.shutdown(wait=True)


class MCTSNode:
    """Node in the MCTS tree for mathematical reasoning with feedback integration"""
    
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
        
        # Manish: added feedback-aware node properties
        self.feedback_history: List[Dict[str, Any]] = []  # Track feedback received
        self.feedback_reward: float = 0.0  # Cumulative feedback reward
        self.feedback_type: Optional[str] = None  # Type of feedback received
        self.last_feedback_info: Optional[Dict[str, Any]] = None  # Latest feedback details
    
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
    
    # Manish: added feedback-aware update methods
    def update_with_feedback(self, value: float, feedback_reward: float, feedback_info: Dict[str, Any]):
        """Update node statistics with feedback information"""
        self.visits += 1
        self.value += value
        self.feedback_reward += feedback_reward
        self.feedback_type = feedback_info.get("feedback_type", "unknown")
        self.last_feedback_info = feedback_info
        self.feedback_history.append({
            "iteration": self.visits,
            "value": value,
            "feedback_reward": feedback_reward,
            "feedback_type": self.feedback_type,
            "timestamp": time.time()
        })
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback received by this node"""
        if not self.feedback_history:
            return {"total_feedback": 0, "feedback_types": [], "avg_feedback_reward": 0.0}
        
        feedback_types = [f.get("feedback_type", "unknown") for f in self.feedback_history]
        avg_feedback_reward = sum(f.get("feedback_reward", 0) for f in self.feedback_history) / len(self.feedback_history)
        
        return {
            "total_feedback": len(self.feedback_history),
            "feedback_types": list(set(feedback_types)),
            "avg_feedback_reward": avg_feedback_reward,
            "total_feedback_reward": self.feedback_reward,
            "latest_feedback": self.last_feedback_info
        }
    
    def get_path(self) -> List[Tuple[Optional[ReasoningAction], ReasoningState]]:
        """Get the path from root to this node"""
        path = []
        node = self
        while node is not None:
            path.append((node.action, node.state))
            node = node.parent
        return list(reversed(path))


class ParallelMCTSReasoningSolver:
    """MCTS-based mathematical reasoning solver with parallel expansion and feedback integration"""
    
    def __init__(self, llm_reasoner: ParallelLLMReasoner, max_iterations: int = 100, 
                 exploration_constant: float = 1.414, max_depth: int = 20,
                 logger: Optional[DebugLogger] = None, parallel_expansions: int = 3,
                 feedback_interface: Optional[FeedbackInterface] = None):
        self.llm_reasoner = llm_reasoner
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.stats = defaultdict(int)
        self.logger = logger or DebugLogger()
        self.parallel_expansions = parallel_expansions
        
        # Manish: added feedback integration
        self.feedback_interface = feedback_interface
        self.feedback_stats = defaultdict(int)  # Track feedback usage statistics
    
    def solve(self, problem: MathProblem) -> Dict[str, Any]:
        """Solve a mathematical problem using MCTS + LLM with parallel expansions"""
        # Initialize root state
        initial_state = ReasoningState(problem=problem)
        root = MCTSNode(initial_state)
        
        best_solution = None
        best_value = -float('inf')
        
        self.logger.log(f"\nStarting MCTS for {problem.subject} problem (Level {problem.level})", 
                       Fore.CYAN, important=True)
        self.logger.log(f"Problem: {problem.problem_text[:200]}...", Fore.WHITE)
        
        iteration = 0
        while iteration < self.max_iterations:
            # Determine how many parallel expansions to do
            num_parallel = min(self.parallel_expansions, self.max_iterations - iteration)
            
            if num_parallel > 0:
                # Collect nodes for parallel expansion
                nodes_to_expand = []
                for _ in range(num_parallel):
                    node = self._select(root)
                    if not node.is_terminal and len(node.get_path()) < self.max_depth:
                        nodes_to_expand.append(node)
                
                if nodes_to_expand:
                    # Parallel expansion and evaluation
                    expanded_nodes = self._expand_parallel(nodes_to_expand)
                    values = self._simulate_parallel(expanded_nodes)
                    
                    # Backpropagate all results
                    for node, value in zip(expanded_nodes, values):
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
                                    "iterations": iteration + num_parallel,
                                    "techniques_used": node.state.techniques_used,
                                    "reasoning_steps": [a.description for a, _ in node.get_path() if a]
                                }
                                
                                if best_solution.get("is_correct") and best_value > 0.95:
                                    self.logger.log(f"\n✅ CORRECT SOLUTION FOUND! Stopping early.", 
                                                  Fore.GREEN, important=True)
                                    iteration = self.max_iterations  # Exit loop
                                    break
                
                iteration += num_parallel
            else:
                break
            
            # Log progress periodically
            if iteration % 10 == 0 or iteration < 5:
                self.logger.log_iteration(iteration, self.max_iterations)
        
        # If no complete solution found, return the best partial solution
        if not best_solution:
            best_node = max(root.children, key=lambda n: n.value / n.visits if n.visits > 0 else 0) if root.children else root
            evaluation = self.llm_reasoner.evaluate_state(best_node.state)
            best_solution = {
                "solution": evaluation.get("solution", "No complete solution found"),
                "is_correct": False,
                "reasoning_path": [(a.description if a else "Initial state", s) 
                                 for a, s in best_node.get_path()],
                "value": best_node.value / best_node.visits if best_node.visits > 0 else 0,
                "confidence": evaluation.get("confidence", 0),
                "iterations": self.max_iterations,
                "techniques_used": best_node.state.techniques_used,
                "reasoning_steps": [a.description for a, _ in best_node.get_path() if a]
            }
        
        best_solution["stats"] = dict(self.stats)
        return best_solution
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCB1"""
        while node.children and node.is_fully_expanded() and not node.is_terminal:
            node = node.best_child(self.exploration_constant)
            self.stats['selections'] += 1
        return node
    
    def _expand_parallel(self, nodes: List[MCTSNode]) -> List[MCTSNode]:
        """Expansion phase for multiple nodes in parallel with feedback integration"""
        # Generate actions for all nodes that need them
        nodes_needing_actions = []
        for node in nodes:
            if not node.untried_actions and not node.is_terminal:
                nodes_needing_actions.append(node)
        
        if nodes_needing_actions:
            # Generate actions in parallel
            states = [node.state for node in nodes_needing_actions]
            all_actions = self.llm_reasoner.generate_actions_parallel(states, num_actions=3)
            
            # Assign actions to nodes
            for node, actions in zip(nodes_needing_actions, all_actions):
                node.untried_actions = sorted(actions, key=lambda a: a.confidence, reverse=True)
        
        # Now expand each node with feedback integration
        expanded_nodes = []
        for node in nodes:
            if node.untried_actions:
                action = node.untried_actions.pop(0)
                
                # Manish: added feedback-aware expansion
                if self.feedback_interface:
                    # Get feedback for the action
                    next_state, feedback_reward, feedback_info = self.feedback_interface.get_feedback(node.state, action)
                    
                    # Create child with feedback-informed state
                    child = node.add_child(next_state, action)
                    
                    # Store feedback information in the child node
                    child.feedback_reward = feedback_reward
                    child.feedback_type = feedback_info.get("feedback_type", "unknown")
                    child.last_feedback_info = feedback_info
                    child.feedback_history.append({
                        "iteration": 1,
                        "value": 0.0,  # Will be updated in simulation
                        "feedback_reward": feedback_reward,
                        "feedback_type": child.feedback_type,
                        "timestamp": time.time()
                    })
                    
                    # Update feedback statistics
                    self.feedback_stats[f"feedback_{child.feedback_type}"] += 1
                    self.feedback_stats["total_feedback_calls"] += 1
                    
                    self.logger.log(f"Feedback expansion: {child.feedback_type} reward={feedback_reward:.3f}", 
                                  Fore.CYAN)
                else:
                    # Fallback to original expansion without feedback
                    new_state = self.llm_reasoner.apply_action(node.state, action)
                    child = node.add_child(new_state, action)
                
                self.stats['expansions'] += 1
                expanded_nodes.append(child)
            else:
                expanded_nodes.append(node)
        
        return expanded_nodes
    
    def _simulate_parallel(self, nodes: List[MCTSNode]) -> List[float]:
        """Simulation phase for multiple nodes in parallel"""
        # Collect nodes that need evaluation
        nodes_to_evaluate = []
        cached_values = []
        
        for i, node in enumerate(nodes):
            if node.evaluation_cache:
                # Use cached evaluation
                evaluation = node.evaluation_cache
            else:
                nodes_to_evaluate.append((i, node))
        
        # Evaluate nodes in parallel
        if nodes_to_evaluate:
            states = [node.state for _, node in nodes_to_evaluate]
            evaluations = self.llm_reasoner.evaluate_states_parallel(states)
            
            # Cache evaluations and mark terminals
            for (i, node), evaluation in zip(nodes_to_evaluate, evaluations):
                node.evaluation_cache = evaluation
                node.is_terminal = evaluation.get("is_complete", False)
        
        # Calculate values for all nodes
        values = []
        for node in nodes:
            evaluation = node.evaluation_cache
            
            if evaluation.get("is_complete") and evaluation.get("is_correct"):
                value = 1.0
            elif evaluation.get("is_complete"):
                value = 0.2
            else:
                progress = evaluation.get("progress", 0.0)
                quality = evaluation.get("quality", 0.0)
                confidence = evaluation.get("confidence", 0.5)
                value = 0.4 * progress + 0.4 * quality + 0.2 * confidence
            
            values.append(value)
            self.stats['simulations'] += 1
        
        return values
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagation phase: update statistics with feedback integration"""
        while node is not None:
            # Manish: added feedback-aware backpropagation
            if hasattr(node, 'feedback_reward') and node.feedback_reward != 0:
                # Use feedback-informed update
                feedback_info = node.last_feedback_info or {"feedback_type": "unknown"}
                node.update_with_feedback(value, node.feedback_reward, feedback_info)
                
                # Log feedback backpropagation
                self.logger.log(f"Feedback backprop: {node.feedback_type} reward={node.feedback_reward:.3f}, value={value:.3f}", 
                              Fore.MAGENTA)
            else:
                # Standard update without feedback
                node.update(value)
            
            node = node.parent
            self.stats['backpropagations'] += 1


class MathReasoningSystemParallel:
    """Complete system for mathematical reasoning with MCTS + LLM, parallelization, and feedback integration"""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7, 
                 verbose_logging: bool = True, max_workers: int = 3,
                 parallel_expansions: int = 3, enable_feedback: bool = True,
                 feedback_type: str = "neural", lean_server_url: str = "http://localhost:8000",
                 neural_weight: float = 0.7, symbolic_weight: float = 0.3):
        self.logger = DebugLogger(verbose=verbose_logging)
        self.llm_reasoner = ParallelLLMReasoner(
            model_name=model_name, 
            temperature=temperature,
            logger=self.logger,
            max_workers=max_workers
        )
        
        # Manish: added feedback system initialization
        self.feedback_interface = None
        if enable_feedback:
            if feedback_type == "neural":
                self.feedback_interface = NeuralFeedbackModule(self.llm_reasoner, self.logger)
            elif feedback_type == "symbolic":
                self.feedback_interface = SymbolicFeedbackModule(lean_server_url, self.logger)
            elif feedback_type == "combined":
                neural_feedback = NeuralFeedbackModule(self.llm_reasoner, self.logger)
                symbolic_feedback = SymbolicFeedbackModule(lean_server_url, self.logger)
                self.feedback_interface = CombinedFeedbackModule(
                    neural_feedback, symbolic_feedback, 
                    neural_weight, symbolic_weight, self.logger
                )
            else:
                self.logger.log(f"Unknown feedback type: {feedback_type}, using neural", Fore.YELLOW)
                self.feedback_interface = NeuralFeedbackModule(self.llm_reasoner, self.logger)
        
        self.solver = ParallelMCTSReasoningSolver(
            llm_reasoner=self.llm_reasoner,
            max_iterations=50,
            exploration_constant=1.414,
            max_depth=15,
            logger=self.logger,
            parallel_expansions=parallel_expansions,
            feedback_interface=self.feedback_interface
        )
        self.results_data = []
        self.results_filename = self.get_results_filename()
        self._results_lock = threading.Lock()  # Thread safety for results
        
        # Log feedback configuration
        if self.feedback_interface:
            self.logger.log(f"Feedback system enabled: {feedback_type}", Fore.GREEN, important=True)
        else:
            self.logger.log("Feedback system disabled", Fore.YELLOW, important=True)
    
    def solve_problem(self, problem: Union[str, MathProblem, Dict[str, Any]], 
                     problem_idx: int = None, total_problems: int = None,
                     dataset_idx: int = None) -> Dict[str, Any]:
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
            try:
                math_problem = MathProblem.from_dataset(problem)
            except Exception as e:
                # If parsing fails, create a basic problem with available data
                self.logger.log(f"Error parsing problem data: {e}", Fore.RED)
                math_problem = MathProblem(
                    problem_text=problem.get('problem', 'Unknown problem'),
                    solution=problem.get('solution', ''),
                    answer=str(problem.get('answer', '')),
                    subject=problem.get('subject', 'Unknown'),
                    level=problem.get('level', 1)
                )
        else:
            math_problem = problem
        
        # Log problem start
        if problem_idx and total_problems:
            self.logger.log_problem_start(problem_idx, total_problems, 
                                        math_problem.subject, math_problem.level)
        
        try:
            result = self.solver.solve(math_problem)
            result['time_taken'] = time.time() - start_time
            result['problem_info'] = {
                'subject': math_problem.subject,
                'level': math_problem.level,
                'has_diagram': math_problem.has_diagram
            }
            
            # Create statistics record with feedback information
            stats_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'dataset_index': dataset_idx,
                'problem_text': math_problem.problem_text[:200] + "..." if len(math_problem.problem_text) > 200 else math_problem.problem_text,
                'subject': math_problem.subject,
                'level': math_problem.level,
                'time_taken': result['time_taken'],
                'iterations': result.get('iterations', 0),
                'reasoning_steps': result.get('reasoning_steps', []),
                'num_reasoning_steps': len(result.get('reasoning_steps', [])),
                'techniques_used': result.get('techniques_used', []),
                'predicted_answer': result.get('solution'),
                'groundtruth_answer': math_problem.answer,  # This now contains the extracted answer
                'groundtruth_solution': math_problem.solution[:500] + "..." if len(math_problem.solution) > 500 else math_problem.solution,
                'is_correct': result.get('is_correct', False),
                'confidence': result.get('confidence', 0),
                'value': result.get('value', 0),
                'has_diagram': math_problem.has_diagram,
                'mcts_stats': result.get('stats', {}),
                # Manish: added feedback statistics
                'feedback_enabled': self.feedback_interface is not None,
                'feedback_stats': dict(self.solver.feedback_stats) if hasattr(self.solver, 'feedback_stats') else {},
                'feedback_type': type(self.feedback_interface).__name__ if self.feedback_interface else 'None'
            }
            
            with self._results_lock:
                self.results_data.append(stats_record)
            
            # Save results incrementally after each problem
            self.save_results_to_json()
            
            # Log result
            self.logger.log(f"\n📊 Result: {result.get('solution')}", Fore.CYAN, important=True)
            self.logger.log(f"Correct: {result.get('is_correct')}", 
                          Fore.GREEN if result.get('is_correct') else Fore.RED, important=True)
            self.logger.log(f"Time taken: {result['time_taken']:.2f} seconds", Fore.BLUE)
            
            return result
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            error_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'dataset_index': dataset_idx,
                'problem_text': math_problem.problem_text[:200] + "...",
                'subject': math_problem.subject,
                'level': math_problem.level,
                'time_taken': time.time() - start_time,
                'error': str(e),
                'is_correct': False
            }
            
            with self._results_lock:
                self.results_data.append(error_record)
            
            # Save results incrementally even for errors
            self.save_results_to_json()
            
            return {
                "solution": None,
                "error": str(e),
                "time_taken": time.time() - start_time
            }
    
    def evaluate_all_subjects_and_levels(self, dataset_name: str = "HuggingFaceH4/MATH-500",
                                       split: str = "test",
                                       problems_per_category: int = 5) -> Dict[str, Any]:
        """Evaluate on all subjects and levels from MATH dataset"""
        
        # All subjects from MATH-500
        all_subjects = ['Precalculus', 'Intermediate Algebra', 'Algebra', 'Number Theory',
                       'Prealgebra', 'Geometry', 'Counting & Probability']
        all_levels = [1, 2, 3, 4, 5]
        
        # Load dataset
        self.logger.log(f"\n{'='*80}", Fore.CYAN, important=True)
        self.logger.log(f"Loading dataset: {dataset_name}", Fore.CYAN, important=True)
        self.logger.log(f"{'='*80}", Fore.CYAN, important=True)
        
        dataset = load_dataset(dataset_name, split=split)
        
        # Group problems by subject and level
        problems_by_category = defaultdict(list)
        for idx, item in enumerate(dataset):
            subject = item.get('subject', 'Unknown')
            level = item.get('level', 0)
            if subject in all_subjects and level in all_levels:
                problems_by_category[(subject, level)].append((idx, item))
        
        # Statistics tracking
        results_by_subject = defaultdict(lambda: {"correct": 0, "total": 0, "by_level": defaultdict(lambda: {"correct": 0, "total": 0})})
        overall_stats = {"correct": 0, "total": 0}
        
        # Calculate total problems to solve
        total_categories = len(all_subjects) * len(all_levels)
        total_problems_to_solve = min(problems_per_category * total_categories,
                                    sum(len(probs) for probs in problems_by_category.values()))
        
        problem_counter = 0
        
        # Evaluate each subject-level combination
        for subject in all_subjects:
            self.logger.log(f"\n{'='*80}", Fore.YELLOW, important=True)
            self.logger.log(f"SUBJECT: {subject}", Fore.YELLOW, important=True)
            self.logger.log(f"{'='*80}", Fore.YELLOW, important=True)
            
            for level in all_levels:
                category_problems = problems_by_category.get((subject, level), [])
                
                if not category_problems:
                    self.logger.log(f"\nNo problems found for {subject} Level {level}", Fore.RED)
                    continue
                
                self.logger.log(f"\n{'-'*60}", Fore.BLUE)
                self.logger.log(f"Evaluating {subject} - Level {level}", Fore.BLUE, important=True)
                self.logger.log(f"Available problems: {len(category_problems)}", Fore.BLUE)
                self.logger.log(f"{'-'*60}", Fore.BLUE)
                
                # Sample problems for this category
                num_to_sample = min(problems_per_category, len(category_problems))
                sampled_indices = np.random.choice(len(category_problems), num_to_sample, replace=False)
                
                for sample_idx in sampled_indices:
                    idx, problem_data = category_problems[sample_idx]
                    problem_counter += 1
                    
                    try:
                        # Solve the problem
                        result = self.solve_problem(problem_data, problem_counter, total_problems_to_solve, 
                                                   dataset_idx=idx)
                        
                        # Update statistics
                        is_correct = result.get("is_correct", False)
                        
                        results_by_subject[subject]["total"] += 1
                        results_by_subject[subject]["by_level"][level]["total"] += 1
                        overall_stats["total"] += 1
                        
                        if is_correct:
                            results_by_subject[subject]["correct"] += 1
                            results_by_subject[subject]["by_level"][level]["correct"] += 1
                            overall_stats["correct"] += 1
                        
                        # Log current accuracy and saved file
                        current_accuracy = overall_stats["correct"] / overall_stats["total"]
                        self.logger.log(f"\nRunning accuracy: {overall_stats['correct']}/{overall_stats['total']} = {current_accuracy:.2%}", 
                                      Fore.CYAN, important=True)
                        self.logger.log(f"Results saved to: {self.results_filename}", Fore.BLUE)
                    except Exception as e:
                        self.logger.log(f"\nError solving problem {idx}: {e}", Fore.RED)
                        self.logger.log("Continuing with next problem...", Fore.YELLOW)
                        continue
        
        # Display overall statistics
        self.display_overall_statistics(results_by_subject, overall_stats)
        
        return {
            "overall_stats": overall_stats,
            "results_by_subject": dict(results_by_subject),
            "num_problems_evaluated": problem_counter,
            "results_file": self.results_filename
        }
    
    def save_results_to_json(self):
        """Save all results to a JSON file (thread-safe)"""
        with self._results_lock:
            summary = {
                "evaluation_date": datetime.datetime.now().isoformat(),
                "total_problems": len(self.results_data),
                "model_config": {
                    "model_name": self.llm_reasoner.llm.model_name,
                    "temperature": self.llm_reasoner.llm.temperature,
                    "max_iterations": self.solver.max_iterations,
                    "exploration_constant": self.solver.exploration_constant,
                    "max_depth": self.solver.max_depth,
                    "parallel_expansions": self.solver.parallel_expansions,
                    "max_workers": self.llm_reasoner.executor._max_workers
                },
                "problems": self.results_data
            }
            
            with open(self.results_filename, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Only log the first time
            if len(self.results_data) == 1:
                self.logger.log(f"\n💾 Results file created: {self.results_filename}", Fore.GREEN, important=True)
    
    def get_results_filename(self):
        """Generate filename for results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"mcts_math_results_parallel_{timestamp}.json"
    
    def display_overall_statistics(self, results_by_subject: Dict, overall_stats: Dict):
        """Display comprehensive statistics"""
        
        self.logger.log(f"\n{'='*80}", Fore.CYAN, important=True)
        self.logger.log("OVERALL STATISTICS", Fore.CYAN, important=True)
        self.logger.log(f"{'='*80}", Fore.CYAN, important=True)
        
        # Overall accuracy
        overall_accuracy = overall_stats["correct"] / overall_stats["total"] if overall_stats["total"] > 0 else 0
        self.logger.log(f"\nTotal Problems: {overall_stats['total']}", Fore.WHITE)
        self.logger.log(f"Correct: {overall_stats['correct']}", Fore.GREEN)
        self.logger.log(f"Overall Accuracy: {overall_accuracy:.2%}", Fore.CYAN, important=True)
        
        # Subject-wise statistics
        self.logger.log(f"\n{'-'*80}", Fore.BLUE)
        self.logger.log("SUBJECT-WISE PERFORMANCE", Fore.BLUE, important=True)
        self.logger.log(f"{'-'*80}", Fore.BLUE)
        
        # Create a sorted list of subjects by accuracy
        subject_accuracies = []
        for subject, stats in results_by_subject.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                subject_accuracies.append((subject, accuracy, stats))
        
        subject_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        # Display subject performance
        for subject, accuracy, stats in subject_accuracies:
            self.logger.log(f"\n{subject}:", Fore.YELLOW, important=True)
            self.logger.log(f"  Overall: {stats['correct']}/{stats['total']} = {accuracy:.2%}", 
                          Fore.GREEN if accuracy >= 0.5 else Fore.RED)
            
            # Level-wise breakdown
            level_stats = []
            for level in sorted(stats["by_level"].keys()):
                level_data = stats["by_level"][level]
                if level_data["total"] > 0:
                    level_acc = level_data["correct"] / level_data["total"]
                    level_stats.append((level, level_acc, level_data))
            
            if level_stats:
                self.logger.log("  By Level:", Fore.BLUE)
                for level, level_acc, level_data in level_stats:
                    self.logger.log(f"    Level {level}: {level_data['correct']}/{level_data['total']} = {level_acc:.2%}",
                                  Fore.GREEN if level_acc >= 0.5 else Fore.RED)
        
        # Difficulty analysis
        self.logger.log(f"\n{'-'*80}", Fore.BLUE)
        self.logger.log("DIFFICULTY ANALYSIS", Fore.BLUE, important=True)
        self.logger.log(f"{'-'*80}", Fore.BLUE)
        
        level_aggregated = defaultdict(lambda: {"correct": 0, "total": 0})
        for subject_stats in results_by_subject.values():
            for level, level_data in subject_stats["by_level"].items():
                level_aggregated[level]["correct"] += level_data["correct"]
                level_aggregated[level]["total"] += level_data["total"]
        
        for level in sorted(level_aggregated.keys()):
            if level_aggregated[level]["total"] > 0:
                level_accuracy = level_aggregated[level]["correct"] / level_aggregated[level]["total"]
                self.logger.log(f"Level {level}: {level_aggregated[level]['correct']}/{level_aggregated[level]['total']} = {level_accuracy:.2%}",
                              Fore.GREEN if level_accuracy >= 0.5 else Fore.RED)
        
        # Time statistics
        if self.results_data:
            avg_time = np.mean([r.get('time_taken', 0) for r in self.results_data])
            avg_steps = np.mean([r.get('num_reasoning_steps', 0) for r in self.results_data])
            avg_iterations = np.mean([r.get('iterations', 0) for r in self.results_data])
            
            self.logger.log(f"\n{'-'*80}", Fore.BLUE)
            self.logger.log("PERFORMANCE METRICS", Fore.BLUE, important=True)
            self.logger.log(f"{'-'*80}", Fore.BLUE)
            self.logger.log(f"Average time per problem: {avg_time:.2f} seconds", Fore.WHITE)
            self.logger.log(f"Average reasoning steps: {avg_steps:.1f}", Fore.WHITE)
            self.logger.log(f"Average MCTS iterations: {avg_iterations:.1f}", Fore.WHITE)
    
    def evaluate_all_problems(self, dataset_name: str = "HuggingFaceH4/MATH-500",
                            split: str = "test") -> Dict[str, Any]:
        """Evaluate on ALL problems from MATH dataset"""
        
        # All subjects from MATH-500
        all_subjects = ['Precalculus', 'Intermediate Algebra', 'Algebra', 'Number Theory',
                       'Prealgebra', 'Geometry', 'Counting & Probability']
        all_levels = [1, 2, 3, 4, 5]
        
        # Load dataset
        self.logger.log(f"\n{'='*80}", Fore.CYAN, important=True)
        self.logger.log(f"Loading FULL dataset: {dataset_name}", Fore.CYAN, important=True)
        self.logger.log(f"{'='*80}", Fore.CYAN, important=True)
        
        dataset = load_dataset(dataset_name, split=split)
        total_problems = len(dataset)
        
        self.logger.log(f"Total problems in dataset: {total_problems}", Fore.YELLOW, important=True)
        
        # Statistics tracking
        results_by_subject = defaultdict(lambda: {"correct": 0, "total": 0, "by_level": defaultdict(lambda: {"correct": 0, "total": 0})})
        overall_stats = {"correct": 0, "total": 0}
        
        # Count problems by category for logging
        problems_by_category = defaultdict(int)
        for item in dataset:
            subject = item.get('subject', 'Unknown')
            level = item.get('level', 0)
            if subject in all_subjects and level in all_levels:
                problems_by_category[(subject, level)] += 1
        
        # Log distribution
        self.logger.log(f"\n{'='*60}", Fore.BLUE)
        self.logger.log("PROBLEM DISTRIBUTION", Fore.BLUE, important=True)
        self.logger.log(f"{'='*60}", Fore.BLUE)
        
        for subject in all_subjects:
            subject_total = sum(problems_by_category.get((subject, level), 0) for level in all_levels)
            if subject_total > 0:
                self.logger.log(f"\n{subject}: {subject_total} problems", Fore.YELLOW)
                for level in all_levels:
                    count = problems_by_category.get((subject, level), 0)
                    if count > 0:
                        self.logger.log(f"  Level {level}: {count} problems", Fore.WHITE)
        
        # Process all problems
        problem_counter = 0
        
        for idx, problem_data in enumerate(dataset):
            problem_counter += 1
            subject = problem_data.get('subject', 'Unknown')
            level = problem_data.get('level', 0)
            
            # Only process problems with valid subject and level
            if subject not in all_subjects or level not in all_levels:
                self.logger.log(f"\nSkipping problem {idx} with invalid subject/level: {subject} Level {level}", Fore.RED)
                continue
            
            try:
                # Solve the problem
                result = self.solve_problem(problem_data, problem_counter, total_problems, dataset_idx=idx)
                
                # Update statistics
                is_correct = result.get("is_correct", False)
                
                results_by_subject[subject]["total"] += 1
                results_by_subject[subject]["by_level"][level]["total"] += 1
                overall_stats["total"] += 1
                
                if is_correct:
                    results_by_subject[subject]["correct"] += 1
                    results_by_subject[subject]["by_level"][level]["correct"] += 1
                    overall_stats["correct"] += 1
                
                # Log current accuracy every 10 problems
                if problem_counter % 10 == 0 or problem_counter == total_problems:
                    current_accuracy = overall_stats["correct"] / overall_stats["total"]
                    self.logger.log(f"\n📊 Progress: {problem_counter}/{total_problems} problems completed", 
                                  Fore.CYAN, important=True)
                    self.logger.log(f"Running accuracy: {overall_stats['correct']}/{overall_stats['total']} = {current_accuracy:.2%}", 
                                  Fore.CYAN, important=True)
                    self.logger.log(f"Results saved to: {self.results_filename}", Fore.BLUE)
                    
                    # Estimate time remaining
                    if problem_counter < total_problems:
                        avg_time = np.mean([r.get('time_taken', 0) for r in self.results_data[-10:]])  # Last 10 problems
                        remaining_problems = total_problems - problem_counter
                        estimated_time = avg_time * remaining_problems / 60  # in minutes
                        self.logger.log(f"Estimated time remaining: {estimated_time:.1f} minutes", Fore.YELLOW)
            except Exception as e:
                self.logger.log(f"\nError solving problem {idx}: {e}", Fore.RED)
                self.logger.log("Continuing with next problem...", Fore.YELLOW)
                # Still update total count
                overall_stats["total"] += 1
                continue
        
        # Final save
        self.save_results_to_json()
        
        # Display overall statistics
        self.display_overall_statistics(results_by_subject, overall_stats)
        
        # Save final summary statistics
        self.save_summary_statistics(results_by_subject, overall_stats)
        
        return {
            "overall_stats": overall_stats,
            "results_by_subject": dict(results_by_subject),
            "num_problems_evaluated": problem_counter,
            "results_file": self.results_filename,
            "summary_file": self.get_summary_filename()
        }
    
    def save_summary_statistics(self, results_by_subject: Dict, overall_stats: Dict):
        """Save summary statistics to a separate JSON file"""
        summary_filename = self.get_summary_filename()
        
        # Calculate detailed statistics
        subject_summaries = {}
        for subject, stats in results_by_subject.items():
            if stats["total"] > 0:
                subject_summary = {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": stats["correct"] / stats["total"],
                    "by_level": {}
                }
                
                for level, level_data in stats["by_level"].items():
                    if level_data["total"] > 0:
                        subject_summary["by_level"][level] = {
                            "total": level_data["total"],
                            "correct": level_data["correct"],
                            "accuracy": level_data["correct"] / level_data["total"]
                        }
                
                subject_summaries[subject] = subject_summary
        
        # Performance metrics
        performance_metrics = {}
        if self.results_data:
            performance_metrics = {
                "average_time_per_problem": np.mean([r.get('time_taken', 0) for r in self.results_data]),
                "average_reasoning_steps": np.mean([r.get('num_reasoning_steps', 0) for r in self.results_data]),
                "average_iterations": np.mean([r.get('iterations', 0) for r in self.results_data]),
                "total_time_minutes": sum([r.get('time_taken', 0) for r in self.results_data]) / 60
            }
        
        summary = {
            "evaluation_date": datetime.datetime.now().isoformat(),
            "dataset": "MATH-500",
            "total_problems": overall_stats["total"],
            "correct": overall_stats["correct"],
            "overall_accuracy": overall_stats["correct"] / overall_stats["total"] if overall_stats["total"] > 0 else 0,
            "subject_performance": subject_summaries,
            "performance_metrics": performance_metrics,
            "model_config": {
                "model_name": self.llm_reasoner.llm.model_name,
                "temperature": self.llm_reasoner.llm.temperature,
                "max_iterations": self.solver.max_iterations,
                "exploration_constant": self.solver.exploration_constant,
                "max_depth": self.solver.max_depth,
                "parallel_expansions": self.solver.parallel_expansions,
                "max_workers": self.llm_reasoner.executor._max_workers
            }
        }
        
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.log(f"\n📊 Summary statistics saved to: {summary_filename}", Fore.GREEN, important=True)
    
    def get_summary_filename(self):
        """Generate filename for summary statistics"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"mcts_math_summary_{timestamp}.json"
    
    def evaluate_on_dataset(self, dataset_name: str = "HuggingFaceH4/MATH-500", 
                          split: str = "test", 
                          num_problems: int = 10,
                          subjects: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate on MATH dataset (backward compatibility)"""
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
        
        for i, idx in enumerate(sample_indices):
            problem_data = dataset[int(idx)]
            
            result = self.solve_problem(problem_data, i + 1, num_problems, 
                                      dataset_idx=int(idx))
            
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
            "average_iterations": np.mean([r.get('iterations', 0) for r in results]),
            "results_file": self.results_filename
        }
    
    def __del__(self):
        """Cleanup: shutdown thread pool"""
        if hasattr(self, 'llm_reasoner'):
            self.llm_reasoner.shutdown()
    
    # Manish: added feedback validation methods
    def validate_feedback_system(self) -> Dict[str, Any]:
        """Validate that the feedback system is working correctly"""
        validation_results = {
            "feedback_enabled": self.feedback_interface is not None,
            "feedback_type": type(self.feedback_interface).__name__ if self.feedback_interface else "None",
            "solver_has_feedback": hasattr(self.solver, 'feedback_interface'),
            "feedback_stats_available": hasattr(self.solver, 'feedback_stats'),
            "validation_timestamp": datetime.datetime.now().isoformat()
        }
        
        if self.feedback_interface:
            # Test feedback with a simple problem
            try:
                test_problem = MathProblem(
                    problem_text="Test problem: 2 + 2 = ?",
                    solution="4",
                    answer="4",
                    subject="Test",
                    level=1
                )
                test_state = ReasoningState(problem=test_problem)
                test_action = ReasoningAction(
                    description="Add 2 + 2",
                    action_type="arithmetic",
                    confidence=0.9
                )
                
                # Test feedback call
                next_state, reward, info = self.feedback_interface.get_feedback(test_state, test_action)
                
                validation_results.update({
                    "feedback_test_successful": True,
                    "test_reward": reward,
                    "test_feedback_type": info.get("feedback_type", "unknown"),
                    "test_state_updated": next_state != test_state
                })
                
            except Exception as e:
                validation_results.update({
                    "feedback_test_successful": False,
                    "test_error": str(e)
                })
        
        return validation_results
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feedback statistics"""
        if not hasattr(self.solver, 'feedback_stats'):
            return {"error": "Feedback statistics not available"}
        
        stats = dict(self.solver.feedback_stats)
        total_feedback = stats.get("total_feedback_calls", 0)
        
        return {
            "total_feedback_calls": total_feedback,
            "feedback_breakdown": {k: v for k, v in stats.items() if k.startswith("feedback_")},
            "feedback_enabled": self.feedback_interface is not None,
            "feedback_type": type(self.feedback_interface).__name__ if self.feedback_interface else "None"
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize system with parallelization and feedback integration
    system = MathReasoningSystemParallel(
        model_name="gpt-4o-mini",  # or "gpt-4" for better performance
        temperature=0.7,
        verbose_logging=True,  # Set to False to reduce output
        max_workers=3,  # Number of parallel LLM calls
        parallel_expansions=3,  # Number of nodes to expand in parallel
        enable_feedback=True,  # Enable feedback-aware reasoning
        feedback_type="neural",  # Options: "neural", "symbolic", "combined"
        neural_weight=0.7,  # Weight for neural feedback (when using combined)
        symbolic_weight=0.3  # Weight for symbolic feedback (when using combined)
    )
    
    # Example 1: Validate feedback system
    print("=== Validating Feedback System ===")
    validation = system.validate_feedback_system()
    print(f"Feedback enabled: {validation.get('feedback_enabled')}")
    print(f"Feedback type: {validation.get('feedback_type')}")
    if validation.get('feedback_test_successful'):
        print(f"Feedback test successful - reward: {validation.get('test_reward')}")
    else:
        print(f"Feedback test failed: {validation.get('test_error', 'Unknown error')}")
    
    # Example 2: Test with a simple problem
    print("\n=== Testing Simple Problem with Feedback ===")
    problem1 = "Solve for x: 2x + 5 = 13"
    result1 = system.solve_problem(problem1, 1, 1, dataset_idx=None)
    print(f"\nSolution: {result1.get('solution')}")
    print(f"Correct: {result1.get('is_correct')}")
    
    # Show feedback statistics
    feedback_stats = system.get_feedback_statistics()
    print(f"\nFeedback Statistics:")
    print(f"Total feedback calls: {feedback_stats.get('total_feedback_calls', 0)}")
    print(f"Feedback breakdown: {feedback_stats.get('feedback_breakdown', {})}")
    
    # Example 2: Evaluate on sampled subjects and levels
    # try:
    #     print("\n=== Example 2: Evaluating Sampled Subjects and Levels ===")
    #     eval_results = system.evaluate_all_subjects_and_levels(
    #         problems_per_category=1  # Solve 1 problem per subject-level combination
    #     )
        
    #     print(f"\nEvaluation complete!")
    #     print(f"Results saved to: {eval_results['results_file']}")
    #     print(f"Total problems evaluated: {eval_results['num_problems_evaluated']}")
    #     print(f"Overall accuracy: {eval_results['overall_stats']['correct']}/{eval_results['overall_stats']['total']} = "
    #           f"{eval_results['overall_stats']['correct']/eval_results['overall_stats']['total']:.2%}")
    # except Exception as e:
    #     print(f"\nError during evaluation: {e}")
    #     print("Check the logs above for details.")
    
    # Example 3: Evaluate on ALL 500 problems
    # try:
    #     print("\n=== Example 3: Evaluating ALL 500 Problems ===")
    #     print("WARNING: This will take several hours to complete!")
    #     print("The system will save progress incrementally, so you can stop and resume if needed.")
        
    #     # Uncomment the following lines to run on all 500 problems
    #     all_results = system.evaluate_all_problems()
        
    #     print(f"\nComplete evaluation finished!")
    #     print(f"Results saved to: {all_results['results_file']}")
    #     print(f"Summary saved to: {all_results['summary_file']}")
    #     print(f"Total problems evaluated: {all_results['num_problems_evaluated']}")
    #     print(f"Overall accuracy: {all_results['overall_stats']['correct']}/{all_results['overall_stats']['total']} = "
    #           f"{all_results['overall_stats']['correct']/all_results['overall_stats']['total']:.2%}")
        
    #     # print("\nTo run on all 500 problems, uncomment the code above.")
    #     # print("Estimated time: 3-5 hours with parallelization")
        
    # except Exception as e:
    #     print(f"\nError during full evaluation: {e}")
    #     print("Check the logs above for details.")
    #     print("Partial results have been saved.")