#!/bin/bash
# ==============================================================================
# Token Economy — Overnight Video Generation
# ==============================================================================
# Usage: ./generate_overnight.sh
#
# This script queues all ready videos (scripts with matching scene plans)
# and runs generation sequentially overnight. Run before bed.
#
# Prerequisites:
#   - Scripts in scripts/ (e.g., 003_why_ai_startups_fail.md)
#   - Scene plans in scripts/ (e.g., 003_why_ai_startups_fail_plan.json)
#   - Virtual env active or use .venv/bin/vidgen directly
# ==============================================================================

set -e

VENV_BIN="$(dirname "$0")/.venv/bin"
VIDGEN="$VENV_BIN/vidgen"
SCRIPTS_DIR="$(dirname "$0")/scripts"

echo "============================================="
echo "  Token Economy — Overnight Generation"
echo "============================================="
echo ""

# Prevent Mac from sleeping
echo "Starting caffeinate (preventing sleep)..."
caffeinate -d -i -s &
CAFFEINE_PID=$!
trap "kill $CAFFEINE_PID 2>/dev/null; echo 'Caffeinate stopped.'" EXIT

# Find all scripts that have matching scene plans and haven't been generated yet
QUEUED=0
for script in "$SCRIPTS_DIR"/*.md; do
    [ -f "$script" ] || continue
    
    # Derive expected plan filename
    base=$(basename "$script" .md)
    plan="$SCRIPTS_DIR/${base}_plan.json"
    
    if [ ! -f "$plan" ]; then
        echo "  SKIP: $base (no scene plan found)"
        continue
    fi
    
    # Check if already generated (look for output directory with matching slug)
    slug=$(grep "topic_slug" "$plan" | head -1 | sed 's/.*: *"//;s/".*//')
    if [ -d "output/"*"_${slug}" ] 2>/dev/null; then
        echo "  DONE: $base (already generated)"
        continue
    fi
    
    echo "  QUEUE: $base"
    "$VIDGEN" queue-add "$script" -p "$plan"
    QUEUED=$((QUEUED + 1))
done

echo ""
if [ $QUEUED -eq 0 ]; then
    echo "Nothing to generate. All videos either done or missing scene plans."
    echo ""
    echo "To prepare videos for generation:"
    echo "  1. Write script: scripts/NNN_topic.md"
    echo "  2. Create plan:  scripts/NNN_topic_plan.json"
    echo "  3. Run this script again."
    exit 0
fi

echo "$QUEUED video(s) queued for generation."
echo "Estimated time: ~$(( QUEUED * 150 / 60 )) hours (at ~2.5 hrs/video current speed)"
echo ""
echo "Starting generation at $(date '+%H:%M:%S')..."
echo "Go to sleep. Check results in the morning."
echo ""

# Run the queue
"$VIDGEN" queue-run

echo ""
echo "============================================="
echo "  Generation complete at $(date '+%H:%M:%S')"
echo "============================================="
echo ""
echo "Output videos in: ./output/"
ls -la output/ 2>/dev/null || echo "(no output yet)"
