# 🚀 Running Full MATH-500 Dataset with Combined System

This guide shows you how to run the complete MATH-500 dataset using the combined neural + LEAN feedback system.

## 📋 Prerequisites

1. **LEAN 4 Server Running**: The real LEAN 4 server must be running
2. **OpenAI API Key**: Set your OpenAI API key for GPT-4o-mini
3. **Sufficient Resources**: Full evaluation takes 4-8 hours

## 🔧 Setup Steps

### 1. Start LEAN 4 Server
```bash
cd /home/manish/research/mcts-reasoning
python real_lean_server.py &
```

### 2. Verify LEAN 4 Server
```bash
curl -X GET http://localhost:8003/health
```

Should return:
```json
{
  "lean_available": true,
  "service": "Real LEAN 4 Server",
  "status": "healthy",
  "version": "4.24.0"
}
```

## 🎯 Running the Full Evaluation

### Option 1: Interactive Mode (Recommended)
```bash
python run_full_math500_combined.py
```
- Asks for confirmation before starting
- Shows detailed progress
- Provides warnings about time/resources

### Option 2: Automated Mode
```bash
python run_math500_auto.py
```
- Runs automatically without confirmation
- Good for scheduled runs
- Less verbose output

## 📊 What to Expect

### Performance Metrics
- **Problems**: ~500 mathematical problems
- **Time**: 4-8 hours total
- **Accuracy**: Expected 75-80% (based on sample results)
- **Subjects**: 7 subjects (Algebra, Geometry, Precalculus, etc.)
- **Levels**: 5 difficulty levels

### Output Files
- **Results**: `mcts_math_results_parallel_YYYYMMDD_HHMMSS.json`
- **Logs**: Console output with progress
- **Statistics**: Subject-wise and level-wise performance

## 🔍 Monitoring Progress

### Real-time Monitoring
```bash
# Check LEAN server status
curl -X GET http://localhost:8003/status

# Monitor system resources
htop

# Check results file (if it exists)
ls -la mcts_math_results_parallel_*.json
```

### Expected Output
```
🚀 Initializing MCTS system with combined feedback...
✅ System initialized with combined feedback
   Neural weight: 70%
   Symbolic weight: 30%
   LEAN server: http://localhost:8003

🔍 Starting full MATH-500 evaluation...
   Dataset: HuggingFaceH4/MATH-500
   Split: test
   Problems per category: ALL (full dataset)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBLEM 1/500 - Algebra (Level 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
...
```

## 📈 Analyzing Results

### After Completion
```bash
# Analyze results
python analyze_results.py

# Compare with previous runs
python analyze_results.py
```

### Key Metrics to Check
1. **Overall Accuracy**: Should be 75-80%
2. **Subject Performance**: All subjects should show improvement
3. **Difficulty Levels**: Higher levels should benefit most
4. **Feedback Statistics**: LEAN integration should be active
5. **Time Efficiency**: Should be faster than neural-only

## 🛠️ Troubleshooting

### Common Issues

#### 1. LEAN Server Not Running
```
❌ LEAN 4 Server connection failed
```
**Solution**: Start the server
```bash
python real_lean_server.py &
```

#### 2. Out of Memory
```
❌ Memory error during evaluation
```
**Solution**: Reduce parallel workers
```python
max_workers=2,  # Instead of 4
parallel_expansions=2,  # Instead of 3
```

#### 3. API Rate Limits
```
❌ OpenAI API rate limit exceeded
```
**Solution**: Add delays or reduce parallel workers

#### 4. LEAN Server Timeout
```
❌ LEAN 4 verification timeout
```
**Solution**: Check server status and restart if needed

### Performance Optimization

#### For Faster Execution
```python
# In the system initialization
max_workers=6,           # More workers
parallel_expansions=4,   # More expansions
max_iterations=30,      # Fewer iterations
```

#### For Better Accuracy
```python
# In the system initialization
max_workers=2,           # Fewer workers (more stable)
parallel_expansions=2,   # Fewer expansions
max_iterations=100,     # More iterations
```

## 📊 Expected Results

Based on the sample evaluation, you should see:

### Overall Performance
- **Accuracy**: 75-80% (vs 60-65% neural-only)
- **Improvement**: +10-15% over neural-only
- **Efficiency**: Faster solving with fewer iterations

### Subject-wise Improvements
- **Algebra**: 100% accuracy
- **Number Theory**: 100% accuracy  
- **Prealgebra**: 100% accuracy
- **Precalculus**: 60-80% accuracy
- **Geometry**: 40-60% accuracy

### Difficulty-wise Improvements
- **Level 1-2**: 70-100% accuracy
- **Level 3**: 80-90% accuracy
- **Level 4-5**: 60-80% accuracy

## 🎯 Success Criteria

The evaluation is successful if:
1. ✅ **Accuracy > 75%** (significant improvement)
2. ✅ **LEAN integration active** (feedback calls > 0)
3. ✅ **All subjects evaluated** (7 subjects)
4. ✅ **All levels evaluated** (5 levels)
5. ✅ **Results file generated** (JSON format)

## 📝 Next Steps After Completion

1. **Analyze Results**: Use `analyze_results.py`
2. **Compare Performance**: Compare with neural-only baseline
3. **Document Findings**: Prepare for research publication
4. **Optimize System**: Fine-tune weights and parameters
5. **Scale Up**: Consider larger datasets or different domains

## 🚨 Important Notes

- **Backup Results**: Save the results file immediately
- **Monitor Resources**: Watch CPU/memory usage
- **Check Logs**: Monitor for errors or warnings
- **Validate LEAN**: Ensure LEAN server stays running
- **Save Progress**: Results are saved incrementally

## 📞 Support

If you encounter issues:
1. Check LEAN server status
2. Verify OpenAI API key
3. Monitor system resources
4. Check error logs
5. Restart if necessary

Good luck with your full MATH-500 evaluation! 🚀
