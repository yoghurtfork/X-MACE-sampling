#!/usr/bin/env bash

echo "traj_00053"
echo $(hostname)



PRIMARY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

cd $PRIMARY_DIR

"$SHARC/sharc.x" input
