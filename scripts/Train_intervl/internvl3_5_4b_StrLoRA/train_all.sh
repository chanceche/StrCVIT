#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_HOME="<CUDA_HOME>"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"



start=1
end=25

for ((i=start; i<=end; i++)); do
    cur=$(printf "data_%03d" "$i")

    echo "======================"
    echo " Training $cur ..."
    echo "======================"
    if [ "$i" -eq 1 ]; then
        bash "$SCRIPT_DIR/train_start.sh" "$cur"
    else
        bash "$SCRIPT_DIR/train_strcvit.sh" "$cur"
    fi

    echo "======================"
    echo " Evaluating $cur ..."
    echo "======================"
    bash <STRCVIT_METHODS_DIR>/scripts/Eval_internvl_proxy/eval_internvl3_5_4b_StrLoRA.sh "$cur"

    echo
done
