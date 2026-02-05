#!/bin/bash
# Ditwah Claims Pipeline Progress Monitor

OUTPUT_FILE="/tmp/claude/-home-ranushi-Taf-claude-sl-newspaper-bias-analysis/tasks/b04a27a.output"

echo "========================================="
echo "Ditwah Claims Pipeline Progress Monitor"
echo "========================================="
echo ""

# Check if pipeline is still running
if pgrep -f "02_generate_claims.py" > /dev/null; then
    echo "Status: ✓ RUNNING"
else
    echo "Status: ⚠ NOT RUNNING (may be complete or stopped)"
fi

echo ""
echo "Latest progress:"
echo "----------------"
tail -15 "$OUTPUT_FILE" | grep -E "INFO.*claim|batch|Stored|Generated|Complete"

echo ""
echo "Summary statistics:"
echo "-------------------"
grep -E "Generated.*claims|Stored.*claims|Found.*articles mentioning" "$OUTPUT_FILE" | tail -5

echo ""
echo "Recent errors (if any):"
echo "-----------------------"
grep -i "error" "$OUTPUT_FILE" | tail -3

echo ""
echo "========================================="
echo "To watch live: tail -f $OUTPUT_FILE"
echo "To stop watching: Press Ctrl+C"
echo "========================================="
