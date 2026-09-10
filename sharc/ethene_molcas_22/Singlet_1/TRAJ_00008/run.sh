#!/usr/bin/env bash

echo "traj_00008"
echo $(hostname)



PRIMARY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

cd $PRIMARY_DIR

"$SHARC/sharc.x" input
