#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Number of times to run
NUM_RUNS=10

echo "Running optimization.py $NUM_RUNS times..."


for i in $(seq 1 $NUM_RUNS); do
    OUTPUT_FILE="$SCRIPT_DIR/optimization_output_$i.txt"
    
    echo "Run $i/$NUM_RUNS..."
    python3 "$SCRIPT_DIR/optimization.py" > "$OUTPUT_FILE" 2>&1
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "  ✓ Run $i completed. Output saved to: optimization_output_$i.txt"
    else
        echo "  ✗ Run $i failed. Check optimization_output_$i.txt for details."
    fi
done

echo "All runs completed!"