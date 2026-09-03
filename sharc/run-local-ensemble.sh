#!/usr/bin/env bash
# Prepare and run SHARC/X-MACE trajectories locally.
# This does not use Slurm or sbatch.

set -euo pipefail

caller_dir=$(pwd -P)

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
  LOCAL_STEPSIZE_FS=<num>  Propagation timestep in fs (default: 0.5).
  LOCAL_NSTATES=<counts>   Optional SHARC state counts, e.g. "2 0 0".
  LOCAL_CHARGE=<charges>   Molecular charges, default: "0 0 0".
  LOCAL_GPU_IDS=<ids>      Comma-separated GPU IDs, e.g. "0,1". Runs one
                           trajectory per listed GPU concurrently.

Without LOCAL_GPU_IDS, trajectories run sequentially. The checkpoint and MACE
device are read from sh-scripts/MACE1.template.
EOF
}

structures=()
reset=false
prepare_only=false
run_existing=false
model_file=""
device="cuda"
energy_unit="eV"
distance_unit="Ang"
tmax_fs="10.0"
stepsize_fs="0.5"
nstates=""
charge="0 0 0"
seed="42"
gpu_ids_value=""

model_file_supplied=false
device_supplied=false
energy_unit_supplied=false
distance_unit_supplied=false
tmax_fs_supplied=false
stepsize_fs_supplied=false
nstates_supplied=false
charge_supplied=false
seed_supplied=false
gpu_ids_supplied=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reset) reset=true; shift ;;
        --prepare-only) prepare_only=true; shift ;;
        --run-existing) run_existing=true; shift ;;
        --model-file)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--model-file requires a value" >&2
                exit 2
            fi
            if [[ "$model_file_supplied" == true ]]; then
                echo "--model-file may only be specified once" >&2
                exit 2
            fi
            model_file=$2; model_file_supplied=true; shift 2 ;;
        --device)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--device requires a value" >&2
                exit 2
            fi
            if [[ "$device_supplied" == true ]]; then
                echo "--device may only be specified once" >&2
                exit 2
            fi
            device=$2; device_supplied=true; shift 2 ;;
        --energy-unit)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--energy-unit requires a value" >&2
                exit 2
            fi
            if [[ "$energy_unit_supplied" == true ]]; then
                echo "--energy-unit may only be specified once" >&2
                exit 2
            fi
            energy_unit=$2; energy_unit_supplied=true; shift 2 ;;
        --distance-unit)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--distance-unit requires a value" >&2
                exit 2
            fi
            if [[ "$distance_unit_supplied" == true ]]; then
                echo "--distance-unit may only be specified once" >&2
                exit 2
            fi
            distance_unit=$2; distance_unit_supplied=true; shift 2 ;;
        --tmax-fs)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--tmax-fs requires a value" >&2
                exit 2
            fi
            if [[ "$tmax_fs_supplied" == true ]]; then
                echo "--tmax-fs may only be specified once" >&2
                exit 2
            fi
            tmax_fs=$2; tmax_fs_supplied=true; shift 2 ;;
        --stepsize-fs)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--stepsize-fs requires a value" >&2
                exit 2
            fi
            if [[ "$stepsize_fs_supplied" == true ]]; then
                echo "--stepsize-fs may only be specified once" >&2
                exit 2
            fi
            stepsize_fs=$2; stepsize_fs_supplied=true; shift 2 ;;
        --nstates)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--nstates requires a value" >&2
                exit 2
            fi
            if [[ "$nstates_supplied" == true ]]; then
                echo "--nstates may only be specified once" >&2
                exit 2
            fi
            nstates=$2; nstates_supplied=true; shift 2 ;;
        --charge)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--charge requires a value" >&2
                exit 2
            fi
            if [[ "$charge_supplied" == true ]]; then
                echo "--charge may only be specified once" >&2
                exit 2
            fi
            charge=$2; charge_supplied=true; shift 2 ;;
        --seed)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--seed requires a value" >&2
                exit 2
            fi
            if [[ "$seed_supplied" == true ]]; then
                echo "--seed may only be specified once" >&2
                exit 2
            fi
            seed=$2; seed_supplied=true; shift 2 ;;
        --gpu-ids)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--gpu-ids requires a value" >&2
                exit 2
            fi
            if [[ "$gpu_ids_supplied" == true ]]; then
                echo "--gpu-ids may only be specified once" >&2
                exit 2
            fi
            gpu_ids_value=$2; gpu_ids_supplied=true; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) structures+=("$1"); shift ;;
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

if [[ "$prepare_only" == true && "$gpu_ids_supplied" == true ]]; then
    echo "--gpu-ids cannot be used with --prepare-only because no trajectories will run." >&2
    exit 2
fi

if [[ "$run_existing" == true ]]; then
    for preparation_setting in \
        model_file device energy_unit distance_unit tmax_fs stepsize_fs nstates charge seed; do
        supplied_name="${preparation_setting}_supplied"
        if [[ "${!supplied_name}" != true ]]; then
            continue
        fi
        option="--${preparation_setting//_/-}"
        if [[ "$option" == "--model-file" ]]; then
            echo "--model-file cannot be used with --run-existing because the MACE" >&2
            echo "configuration is already stored in the prepared ensemble. Rebuild the" >&2
            echo "ensemble with --reset to change it." >&2
        else
            echo "$option cannot be used with --run-existing because the ensemble is already prepared." >&2
        fi
        exit 2
    done
fi

if [[ ${#structures[@]} -eq 0 ]]; then
    usage >&2
    exit 2
fi

if [[ "$run_existing" == false ]]; then
    if [[ "$model_file_supplied" != true ]]; then
        echo "--model-file is required when preparing an ensemble." >&2
        exit 2
    fi
    if [[ "$model_file" != /* ]]; then
        model_file="$caller_dir/$model_file"
    fi
    if [[ ! -f "$model_file" ]]; then
        echo "--model-file must name an existing regular file: $model_file" >&2
        exit 2
    fi
    model_file=$(realpath -e -- "$model_file")
fi

if ! python -c 'import math, sys; value = float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value > 0))' "$tmax_fs"; then
    echo "--tmax-fs must be a finite number greater than zero." >&2
    exit 2
fi
if ! python -c 'import math, sys; step = float(sys.argv[1]); total = float(sys.argv[2]); raise SystemExit(not (math.isfinite(step) and step > 0 and step <= total))' "$stepsize_fs" "$tmax_fs"; then
    echo "--stepsize-fs must be a finite number greater than zero and no greater than --tmax-fs." >&2
    exit 2
fi
if ! python -c 'import random, sys; random.Random(int(sys.argv[1]))' "$seed"; then
    echo "--seed must be an integer accepted by Python random.Random." >&2
    exit 2
fi
if [[ "$nstates_supplied" == true ]] && ! python -c 'import re, sys; raise SystemExit(not bool(re.fullmatch(r"\s*\d+(?:\s+\d+)*\s*", sys.argv[1])))' "$nstates"; then
    echo "--nstates must be whitespace-separated non-negative integers." >&2
    exit 2
fi
if ! python -c 'import re, sys; raise SystemExit(not bool(re.fullmatch(r"\s*[+-]?\d+(?:\s+[+-]?\d+)*\s*", sys.argv[1])))' "$charge"; then
    echo "--charge must be whitespace-separated integers." >&2
    exit 2
fi
for mace_setting in device energy_unit distance_unit; do
    value=${!mace_setting}
    if [[ -z "$value" || "$value" =~ [[:space:]] ]]; then
        echo "--${mace_setting//_/-} must be a non-empty single token." >&2
        exit 2
    fi
done

validated_gpu_ids=()
if [[ "$gpu_ids_supplied" == true ]]; then
    if ! python -c 'import re, sys; value = sys.argv[1]; valid = bool(re.fullmatch(r"\d+(?:,\d+)*", value)); ids = value.split(",") if valid else []; raise SystemExit(not valid or len(ids) != len(set(ids)))' "$gpu_ids_value"; then
        echo "--gpu-ids must be unique comma-separated non-negative integers, e.g. 0,1." >&2
        exit 2
    fi
    IFS=',' read -r -a validated_gpu_ids <<< "$gpu_ids_value"
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

is_cuda_device() {
    [[ "$1" == "cuda" || "$1" == cuda:* ]]
}

if [[ "$gpu_ids_supplied" == true ]]; then
    if [[ "$run_existing" == false ]]; then
        if ! is_cuda_device "$device"; then
            echo "--gpu-ids requires a CUDA-based --device when preparing an ensemble." >&2
            exit 2
        fi
    else
        incompatible_device=false
        for structure in "${structures[@]}"; do
            mace_template="$script_dir/$structure/traj-allmols/QM_shared/MACE.template"
            if [[ ! -f "$mace_template" ]]; then
                echo "Cannot inspect MACE device; missing $mace_template" >&2
                incompatible_device=true
                continue
            fi
            if ! stored_device=$(awk '$1 == "device" { count++; device = $2 } END { if (count != 1 || device == "") exit 1; print device }' "$mace_template"); then
                echo "Cannot read a single device setting from $mace_template" >&2
                incompatible_device=true
                continue
            fi
            if ! is_cuda_device "$stored_device"; then
                echo "--gpu-ids requires a CUDA-based device; $mace_template uses $stored_device." >&2
                incompatible_device=true
            fi
        done
        if [[ "$incompatible_device" == true ]]; then
            exit 2
        fi
    fi
fi

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

for structure in "${structures[@]}"; do
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
            --model-file "$model_file"
            --device "$device"
            --energy-unit "$energy_unit"
            --distance-unit "$distance_unit"
            --charge "$charge"
            --tmax "$tmax_fs"
            --stepsize "$stepsize_fs"
            --seed "$seed"
        )
        if [[ "$nstates_supplied" == true ]]; then
            setup_args+=(--nstates "$nstates")
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
