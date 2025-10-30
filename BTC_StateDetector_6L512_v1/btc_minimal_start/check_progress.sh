#!/bin/bash
# Quick progress checker for running experiments

cd "/Users/mazenlawand/Documents/Caculin ML/btc_minimal_start"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║      INCREMENTAL TRAINING EXPERIMENTS - STATUS             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

check_experiment() {
    local name=$1
    local log_file="logs/${name}.log"
    
    echo "📊 $name"
    echo "   ────────────────────────────────────────"
    
    if [ ! -f "$log_file" ]; then
        echo "   ❓ Not started yet"
        return
    fi
    
    # Check if it's done
    if grep -q "TRAINING COMPLETE" "$log_file"; then
        local accuracy=$(grep -A 3 "TRAINING COMPLETE" "$log_file" | grep "Accuracy:" | awk '{print $2}')
        local features=$(grep -A 3 "TRAINING COMPLETE" "$log_file" | grep "Features:" | awk '{print $2}')
        echo "   ✅ COMPLETE!"
        echo "   Final accuracy: $accuracy"
        echo "   Features used: $features"
    else
        # Check current iteration
        local iteration=$(grep "ITERATION" "$log_file" | tail -1 | awk '{print $2}')
        local current_acc=$(grep "Best:" "$log_file" | tail -1 | grep -oE "[0-9]+\.[0-9]+ accuracy" | awk '{print $1}')
        
        if [ -n "$iteration" ]; then
            echo "   🔄 Running - $iteration"
            if [ -n "$current_acc" ]; then
                echo "   Current best: $current_acc"
            fi
        else
            echo "   ⏳ Initializing (feature engineering)..."
        fi
    fi
    echo ""
}

check_experiment "exp1_derivatives"
check_experiment "exp2_volatility"
check_experiment "exp3_advanced"

echo "───────────────────────────────────────────────────"
echo "💡 Run this script again to check progress"
echo "   Or check individual logs: logs/exp*.log"



