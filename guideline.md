### 🎯 **Goal**

Incorporate **neural and symbolic feedback** into `mcts_parallel.py` so that the MCTS loop adjusts tate expansion, pruning, and value updates based on structured feedback (without using any neural network policy/value model).

### 🧠 **Detailed Implementation Plan**

1. **Define Feedback Interface (`FeedbackInterface` class)**

   * Add a small interface or abstract class within or alongside this file that defines:

     ```python
     class FeedbackInterface:
         def get_feedback(self, state, action):
             """Return (next_state, reward, info)."""
             raise NotImplementedError
     ```
   * It should support:

     * **Neural feedback:** numeric value in [−1, 1].
     * **Symbolic feedback (LEAN):**
       Input → `(proof_state, proposed_tactic)`
       Output → `(new_proof_state, +r_small)` if success, or `(error_message, −r_large)` if failure.

2. **Add a `SymbolicFeedbackModule`**

   * Create or import a small module/class (e.g., `symbolic_feedback.py`) that wraps interaction with a LEAN/Kimina server.
   * Responsibilities:

     * Maintain a persistent LEAN process.
     * Given `(proof_state, tactic)`, run it through the server.
     * Return structured feedback in the format above.

3. **Integrate Feedback into MCTS Rollout**

   * In your rollout or simulation function inside `mcts_parallel.py`, after performing an action or expanding a node:

     * Call the `FeedbackInterface` (either symbolic or neural).
     * Store the returned reward and updated state in the node.
   * Example:

     ```python
     next_state, reward, info = feedback_interface.get_feedback(state, action)
     node.update(reward)
     ```

4. **Modify Backpropagation**

   * Replace learned value network updates with lightweight feedback updates:

     ```python
     def backpropagate(self, node, reward):
         while node is not None:
             node.visits += 1
             node.value += (reward - node.value) / node.visits
             node = node.parent
     ```
   * For symbolic runs:

     * Success → `reward = +1`
     * Failed tactic → `reward = -1`
     * Partial progress → small positive reward (e.g., +0.1)

5. **Parallel Integration**

   * Make sure the feedback calls are thread-safe and don’t block other workers.
   * Each worker process should maintain its own feedback interface or client connection.

6. **Logging and Visualization**

   * Extend your existing logging to record:

     * Feedback type (neural/symbolic)
     * Reward value
     * Updated node value
   * Add optional visualization hooks (e.g., matplotlib or wandb) to track node-value evolution and reward distribution.

7. **Sanity Tests**

   * Run small-scale experiments (10–20 problems) to confirm:

     * LEAN feedback correctly modifies exploration.
     * Reward propagation behaves stably.
     * Parallel workers don’t hang or desync.

8. **Iteration & Tuning**

   * Add a weighting hyperparameter:

     ```python
     total_reward = w_neural * neural_feedback + w_symbolic * symbolic_feedback
     ```
   * Tune `w_neural` and `w_symbolic` in config or CLI args.

### 🪶 **Coding Style Notes**

* Keep all additions clearly commented using:

  ```python
  # Manish: added feedback-aware update logic
  ```
* Don’t remove existing functionality — extend the current parallel MCTS pipeline.
* Keep all imports and function names stable.
* If there is deprecated or new imports or usage follow those best practices and updates.

### ✅ **Deliverables**

* Updated `mcts_parallel.py` with feedback-aware search.
* (Optional) New helper file `symbolic_feedback.py`.
* Log statements confirming when symbolic/neural feedback alters exploration.
* Brief summary comment at the top of `mcts_parallel.py` describing the new feedback mechanism.
