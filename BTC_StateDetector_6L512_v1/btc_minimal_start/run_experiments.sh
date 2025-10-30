#!/bin/bash
# Run multiple incremental training experiments with different starting features

cd "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start"
source venv/bin/activate

echo "🧪 Running Multiple Incremental Training Experiments"
echo "====================================================="
echo ""

# Experiment 1: Derivative-focused (multi-timeframe trends)
echo "📊 Experiment 1: Multi-timeframe derivatives"
python3 - <<'EOF'
import sys
sys.path.insert(0, "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start")

# Modify starting features for this experiment
starting_features_override = ['deriv30d_roc', 'deriv7d_roc', 'deriv3d_roc']
experiment_name = 'exp1_trend_focus'

# Read and modify the script
with open('incremental_simple.py', 'r') as f:
    script = f.read()

# Replace the starting features check
script = script.replace(
    "available_starting = [f for f in starting_features if f in all_features]",
    f"available_starting = {starting_features_override}"
)

# Execute
exec(script)
EOF

# Save results
if [ -f "results/incremental_final.json" ]; then
    cp results/incremental_final.json results/exp1_results.json
    echo "✅ Experiment 1 complete"
fi

sleep 3

# Experiment 2: Volatility-focused
echo ""
echo "📊 Experiment 2: Multi-timeframe volatility"
python3 - <<'EOF'
import sys
sys.path.insert(0, "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start")

starting_features_override = ['volatility_72', 'volatility_24', 'volatility_12']

with open('incremental_simple.py', 'r') as f:
    script = f.read()

script = script.replace(
    "available_starting = [f for f in starting_features if f in all_features]",
    f"available_starting = {starting_features_override}"
)

exec(script)
EOF

if [ -f "results/incremental_final.json" ]; then
    cp results/incremental_final.json results/exp2_results.json
    echo "✅ Experiment 2 complete"
fi

sleep 3

# Experiment 3: Derivative primes (acceleration)
echo ""
echo "📊 Experiment 3: Derivative primes (acceleration)"
python3 - <<'EOF'
import sys
sys.path.insert(0, "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start")

starting_features_override = ['deriv7d_prime7d', 'deriv24h_prime24h', 'deriv4h_prime2h']

with open('incremental_simple.py', 'r') as f:
    script = f.read()

script = script.replace(
    "available_starting = [f for f in starting_features if f in all_features]",
    f"available_starting = {starting_features_override}"
)

exec(script)
EOF

if [ -f "results/incremental_final.json" ]; then
    cp results/incremental_final.json results/exp3_results.json
    echo "✅ Experiment 3 complete"
fi

echo ""
echo "====================================================="
echo "✅ All experiments complete!"
echo "Check results/ directory for individual results"



