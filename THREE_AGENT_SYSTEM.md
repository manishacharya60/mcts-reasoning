# Three-Agent System for LEAN Integration in MCTS

## Overview

The three-agent system seamlessly integrates LEAN verifier into MCTS for mathematical reasoning. It consists of three specialized agents that work together to provide structured feedback that guides MCTS exploration and value updates.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    THREE-AGENT SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   LEAN       │───▶│   PROOF      │───▶│  REFLECTOR   │    │
│  │  GENERATOR   │    │  VALIDATOR   │    │   AGENT      │    │
│  │   (LLM)      │    │  (LEAN)      │    │ (LLM + LEAN) │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                   │                   │             │
│         │                   │                   │             │
│         ▼                   ▼                   ▼             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              MCTS VALUE FUNCTION UPDATE                 │ │
│  │         (Numerical Signal from Reflector)              │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. LEAN Generator Agent

**Purpose**: Generate LEAN proof sketches/subgoals as initial sketches or LEAN code as subgoals in MCTS.

**Implementation**:
- Uses LLM to convert mathematical reasoning steps into LEAN 4 proof sketches
- Generates syntactically correct LEAN code
- Creates subgoal descriptions for tracking progress

**Output**:
```json
{
    "lean_code": "theorem step_name : ... := by ...",
    "proof_sketch": "Natural language description",
    "subgoal_description": "What this subgoal proves",
    "tactics_used": ["tactic1", "tactic2"]
}
```

### 2. Proof Validator Agent

**Purpose**: Validate if the generated subgoal has successfully solved a part of the proof.

**Implementation**:
- Sends LEAN code to LEAN server for verification
- Validates proof syntax and semantics
- Returns success/failure status and proof progress

**Output**:
```json
{
    "success": true/false,
    "proof_progress": 0.0-1.0,
    "error": "Error message if failed"
}
```

### 3. Reflector Agent

**Purpose**: Reflect on the current step of generator and validator to see if everything goes well.

**Implementation**:
- Combines LLM reasoning with LEAN symbolic feedback
- Analyzes quality, correctness, and progress
- Produces numerical signal for MCTS value updates
- Provides recommendations (continue/prune/explore)

**Output**:
```json
{
    "reflection": "Detailed analysis",
    "quality_score": 0.0-1.0,
    "correctness_score": 0.0-1.0,
    "progress_score": 0.0-1.0,
    "numerical_signal": -1.0 to 1.0,
    "recommendation": "continue/prune/explore"
}
```

## Integration with MCTS

### Expansion Phase

When a new node is expanded in MCTS:

1. **Generator** creates LEAN proof sketch for the reasoning action
2. **Validator** checks if the subgoal is valid via LEAN server
3. **Reflector** analyzes both outputs and produces numerical signal
4. **Child Node** is created with feedback information stored

### Backpropagation Phase

The numerical signal from the Reflector is used to update MCTS value function:

- **Combined Value**: 70% reflector signal + 30% simulation value
- **Pruning**: If recommendation is "prune" and signal < -0.3, node exploration is reduced
- **Value Updates**: All parent nodes are updated with feedback-informed values

## Usage

### Basic Usage

```python
from mcts_parallel import MathReasoningSystemParallel, MathProblem

# Create system with three-agent feedback
system = MathReasoningSystemParallel(
    model_name="gpt-4o-mini",
    temperature=0.7,
    enable_feedback=True,
    feedback_type="three_agent",  # Use three-agent system
    lean_server_url="http://localhost:8000"
)

# Solve a problem
problem = MathProblem(
    problem_text="Your problem here",
    solution="Solution",
    answer="Answer",
    subject="Algebra",
    level=1
)

result = system.solve_problem(problem)
```

### Configuration

The three-agent system is configured through the `MathReasoningSystemParallel` constructor:

- `feedback_type="three_agent"`: Enable three-agent system
- `lean_server_url`: URL of LEAN 4 server (default: "http://localhost:8000")
- All other parameters work as before (model_name, temperature, etc.)

## Flow Diagram

```
MCTS Expansion
    │
    ▼
┌─────────────────┐
│  Select Action  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generator      │ ────▶ LEAN Proof Sketch
│  (LLM)          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validator      │ ────▶ LEAN Verification
│  (LEAN Server)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reflector      │ ────▶ Numerical Signal
│  (LLM + LEAN)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Create Child   │ ────▶ Store Feedback Info
│  Node           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Backpropagate  │ ────▶ Update Value Function
│  (Use Signal)   │
└─────────────────┘
```

## Key Features

1. **Seamless Integration**: Three agents work together as a unified feedback system
2. **Structured Feedback**: Each agent provides structured output for the next agent
3. **Numerical Signal**: Reflector produces a single numerical signal for MCTS value updates
4. **Pruning Support**: Reflector recommendations can trigger tree pruning
5. **Comprehensive Logging**: All agent interactions are logged for debugging

## Benefits

1. **Formal Verification**: LEAN provides rigorous mathematical verification
2. **Intelligent Guidance**: Reflector combines neural and symbolic feedback
3. **Efficient Exploration**: Pruning reduces exploration of poor paths
4. **Better Solutions**: Combined feedback improves solution quality

## Testing

Run the test script to see the three-agent system in action:

```bash
python test_three_agent_system.py
```

Make sure the LEAN server is running:

```bash
python real_lean_server.py
```

## Future Enhancements

1. **Parallel Agent Execution**: Run generator and validator in parallel
2. **Adaptive Weights**: Learn optimal weights for combining signals
3. **Advanced Pruning**: More sophisticated pruning strategies
4. **Caching**: Cache LEAN verification results for similar subgoals





