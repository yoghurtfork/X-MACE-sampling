#!/usr/bin/env bash
# Prepare and run SHARC/X-MACE trajectories locally, one at a time.
# This is for small test ensembles; it does not use Slurm or sbatch.

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

The script runs trajectories sequentially in the current shell. The checkpoint
path is read from sh-scripts/MACE1.template.
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

    while IFS= read -r -d '' traj_dir; do
        echo "=== Running ${traj_dir#$script_dir/} ==="
        (
            cd "$traj_dir"
            "$sharc_bin/driver.py" -i mace input > driver.log 2>&1
        ) || {
            echo "Trajectory failed: $traj_dir. Last lines of driver.log:" >&2
            tail -n 40 "$traj_dir/driver.log" >&2 || true
            exit 1
        }
    done < <(find "$work_dir" -type d -name 'TRAJ_*' -print0 | sort -z)

    echo "=== Finished $structure ==="
done
