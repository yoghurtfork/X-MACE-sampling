#!/usr/bin/env bash
# Prepare and run SHARC/X-MACE trajectories locally.
# This does not use Slurm or sbatch.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash run-local-ensemble.sh [--reset] [--prepare-only|--run-existing] structure_0001 [structure_0002 ...]

Options:
  --reset         Delete an existing <structure>/traj-allmols before rebuilding it.
  --prepare-only  Create trajectory folders and inputs, but do not run SHARC.
  --run-existing  Run trajectories already prepared in <structure>/traj-allmols.

Environment:
  SHARC=<path>             Required; SHARC installation directory.
  LOCAL_TMAX_FS=<number>   Test trajectory length in fs (default: 10.0).
  LOCAL_NSTATES=<counts>   Optional SHARC state counts, e.g. "2 0 0".
  LOCAL_CHARGE=<charges>   Molecular charges, default: "0 0 0".
  LOCAL_GPU_IDS=<ids>      Comma-separated GPU IDs, e.g. "0,1". Runs one
                           trajectory per listed GPU concurrently.

Without LOCAL_GPU_IDS, trajectories run sequentially. The checkpoint and MACE
device are read from sh-scripts/MACE1.template.
EOF
}

reset=false
prepare_only=false
run_existing=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reset) reset=true; shift ;;
        --prepare-only) prepare_only=true; shift ;;
        --run-existing) run_existing=true; shift ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) break ;;
    esac
done

if [[ "$prepare_only" == true && "$run_existing" == true ]]; then
    echo "--prepare-only and --run-existing cannot be used together." >&2
    exit 2
fi
if [[ "$reset" == true && "$run_existing" == true ]]; then
    echo "--reset and --run-existing cannot be used together." >&2
    exit 2
fi

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

if [[ -z "${SHARC:-}" ]]; then
    echo "Set SHARC to the SHARC bin directory or installation root." >&2
    exit 1
elif [[ -x "$SHARC/driver.py" ]]; then
    sharc_bin="$SHARC"
    sharc_root=$(cd -- "$SHARC/.." && pwd)
elif [[ -x "$SHARC/bin/driver.py" ]]; then
    sharc_bin="$SHARC/bin"
    sharc_root="$SHARC"
else
    echo "SHARC does not contain driver.py: $SHARC" >&2
    exit 1
fi

# SHARC's Python scripts import modules from <SHARC root>/lib. Normal SHARC
# shell setup does this already; keeping it here makes this local runner work
# from a fresh terminal too. This installation expects $SHARC itself to point
# to bin/, because that is where its interface modules (e.g. SHARC_MACE.py)
# reside.
export SHARC="$sharc_bin"
export PYTHONPATH="$sharc_root/lib${PYTHONPATH:+:$PYTHONPATH}"
# Avoid a Numba cache-location error while setup_traj imports every interface.
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
tmax_fs=${LOCAL_TMAX_FS:-10.0}
gpu_ids=()
if [[ -n "${LOCAL_GPU_IDS:-}" ]]; then
    IFS=',' read -r -a gpu_ids <<< "$LOCAL_GPU_IDS"
    for gpu_id in "${gpu_ids[@]}"; do
        if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
            echo "LOCAL_GPU_IDS must be comma-separated numeric GPU IDs, e.g. 0,1." >&2
            exit 2
        fi
    done
    if ! grep -Eq '^device[[:space:]]+cuda' "$script_dir/sh-scripts/MACE1.template"; then
        echo "GPU scheduling requires 'device cuda' in sh-scripts/MACE1.template." >&2
        exit 1
    fi
fi

run_trajectory() {
    local trajectory_dir=$1
    local gpu_id=${2:-}
    local label=${trajectory_dir#$script_dir/}
    if [[ -n "$gpu_id" ]]; then
        echo "=== Running $label on GPU $gpu_id ==="
    else
        echo "=== Running $label ==="
    fi
    (
        cd "$trajectory_dir"
        if [[ -n "$gpu_id" ]]; then
            CUDA_VISIBLE_DEVICES="$gpu_id" "$sharc_bin/driver.py" -i mace input > driver.log 2>&1
        else
            "$sharc_bin/driver.py" -i mace input > driver.log 2>&1
        fi
    )
}

run_trajectories() {
    local work_dir=$1
    local trajectories=()
    mapfile -d '' trajectories < <(find "$work_dir" -type d -name 'TRAJ_*' -print0 | sort -z)
    if [[ ${#trajectories[@]} -eq 0 ]]; then
        echo "No TRAJ_* directories found in $work_dir." >&2
        return 1
    fi

    if [[ ${#gpu_ids[@]} -eq 0 ]]; then
        local trajectory_dir
        for trajectory_dir in "${trajectories[@]}"; do
            run_trajectory "$trajectory_dir" || return 1
        done
        return 0
    fi

    local next=0 failed=0 gpu_id trajectory_dir pid
    while [[ $next -lt ${#trajectories[@]} ]]; do
        local pids=() labels=()
        for gpu_id in "${gpu_ids[@]}"; do
            [[ $next -lt ${#trajectories[@]} ]] || break
            trajectory_dir=${trajectories[$next]}
            run_trajectory "$trajectory_dir" "$gpu_id" &
            pids+=("$!")
            labels+=("$trajectory_dir")
            ((next += 1))
        done
        for ((i=0; i<${#pids[@]}; i++)); do
            if ! wait "${pids[$i]}"; then
                echo "Trajectory failed: ${labels[$i]}. Last lines of driver.log:" >&2
                tail -n 40 "${labels[$i]}/driver.log" >&2 || true
                failed=1
            fi
        done
        [[ $failed -eq 0 ]] || return 1
    done
}

for structure in "$@"; do
    structure_dir="$script_dir/$structure"
    work_dir="$structure_dir/traj-allmols"

    if [[ ! -f "$structure_dir/initconds.excited" ]]; then
        echo "Missing $structure/initconds.excited; skipping." >&2
        continue
    fi

    if [[ "$run_existing" == false && -e "$work_dir" ]]; then
        if [[ "$reset" != true ]]; then
            echo "$work_dir already exists. Re-run with --reset to replace it." >&2
            exit 1
        fi
        rm -rf -- "$work_dir"
    fi

    if [[ "$run_existing" == false ]]; then
        echo "=== Setting up $structure ==="
        setup_args=(
            --initconds "$structure_dir/initconds.excited"
            --output "$work_dir"
            --template "$script_dir/sh-scripts/MACE1.template"
            --resources "$script_dir/sh-scripts/MACE.resources"
            --charge "${LOCAL_CHARGE:-0 0 0}"
            --tmax "$tmax_fs"
        )
        if [[ -n "${LOCAL_NSTATES:-}" ]]; then
            setup_args+=(--nstates "$LOCAL_NSTATES")
        fi
        python "$script_dir/setup_local_traj.py" "${setup_args[@]}"
    elif [[ ! -d "$work_dir" ]]; then
        echo "No prepared trajectory directory: $work_dir" >&2
        exit 1
    fi

    if [[ "$prepare_only" == true ]]; then
        echo "Prepared $structure; no dynamics requested."
        continue
    fi

    run_trajectories "$work_dir"

    echo "=== Finished $structure ==="
done
