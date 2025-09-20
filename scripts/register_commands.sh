# Configuration for interactive shell working directories
JOZO_WORKDIR="/c/p/assignment1-basics"
POD_WORKDIR="/workspace"
JOZODOC_CONTAINER="assignment1-basics_devcontainer-cs336-dev-1"

function load_runpod_env() {
  # Safely load runpod environment variables without crashing the shell
  local env_vars
  local all_output
  if all_output=$(bash -c '. /workspace/scripts/runpod_environment.sh && echo "RP_IP=$RP_IP" && echo "RP_SSH_PORT=$RP_SSH_PORT" && echo "POD_ID=$POD_ID"' 2>&1); then
    # Extract only the environment variable lines for eval
    env_vars=$(echo "$all_output" | grep -E "^(RP_IP|RP_SSH_PORT|POD_ID)=")
    eval "$env_vars"
    return 0
  else
    echo "Error: Failed to load runpod environment" >&2
    echo "Output from runpod_environment.sh:" >&2
    echo "$all_output" >&2
    return 1
  fi
}

function run_jozo() {
  if [ $# -eq 0 ]; then
    # No arguments - start interactive shell in configured directory
    ssh kruno@jozo -i ~/.ssh/id_jozo -t "cd ${JOZO_WORKDIR} && bash"
  else
    # Execute the provided command
    echo "cd /mnt/c/p/assignment1-basics; $@" | ssh kruno@jozo -i ~/.ssh/id_jozo "bash"
  fi
}

function run_jozodoc() {
  if [ $# -eq 0 ]; then
    # No arguments - start interactive bash shell in the container
    ssh kruno@jozo -i ~/.ssh/id_jozo -t "docker exec -it ${JOZODOC_CONTAINER} bash"
  else
    # Execute the provided command in the container
    echo "cd /mnt/c/p/assignment1-basics; $@" | ssh kruno@jozo -i ~/.ssh/id_jozo -t "docker exec -it ${JOZODOC_CONTAINER} bash"
  fi
}

function run_pod() {
  if load_runpod_env; then
    if [ $# -eq 0 ]; then
      # No arguments - start interactive shell in configured directory
      ssh -i ~/.ssh/id_runpod -p "$RP_SSH_PORT" root@"$RP_IP" -t "cd ${POD_WORKDIR} && bash"
    else
      # Execute the provided command
      echo "cd /mnt/c/p/assignment1-basics; $@" | ssh -i ~/.ssh/id_runpod -t -p "$RP_SSH_PORT" root@"$RP_IP"
    fi
  else
    return 1
  fi
}

function run() {
  local target="$1"
  shift
  local func_name="run_${target}"
  
  if declare -f "$func_name" > /dev/null; then
    "$func_name" "$@"
  else
    # Automatically detect available targets
    local available_targets=$(declare -F | grep "^declare -f run_" | sed 's/declare -f run_//' | tr '\n' ' ')
    echo "Error: Unknown target '$target'. Available targets: ${available_targets}"
    return 1
  fi
}

function push_jozo() {
  . ./scripts/rsync.sh
  # Use -l to preserve symbolic links for jozo
  RSYNC_EXTRA_FLAGS="" RSYNC_RSH="ssh -i ~/.ssh/id_jozo" rsync_default --exclude 'data/*' --rsync-path="wsl rsync" . kruno@jozo:/mnt/c/p/assignment1-basics/
}
function push_pod() {
  if load_runpod_env; then
    . ./scripts/rsync.sh
    # Use -L to follow/convert symbolic links for runpod
    RSYNC_EXTRA_FLAGS="-L" RSYNC_RSH="ssh -i ~/.ssh/id_runpod -p ${RP_SSH_PORT} -o StrictHostKeyChecking=no" rsync_default \
      --include 'data/' \
      --include 'data/owt_train.txt.tokens.vocab_size=32000.npy' \
      --include 'data/owt_valid.txt.tokens.vocab_size=32000.npy' \
      --exclude 'data/*' \
      . root@"${RP_IP}":/workspace
  else
    return 1
  fi
}

function push() {
  local target="$1"
  shift
  local func_name="push_${target}"
  
  if declare -f "$func_name" > /dev/null; then
    "$func_name" "$@"
  else
    # Automatically detect available targets
    local available_targets=$(declare -F | grep "^declare -f push_" | sed 's/declare -f push_//' | tr '\n' ' ')
    echo "Error: Unknown target '$target'. Available targets: ${available_targets}"
    return 1
  fi
}

function rd() {
  local target="$1"
  shift
  
  echo "Push and run on ${target}..."
  if push "$target"; then
    run "$target" "$@"
  else
    echo "Push failed, aborting run"
    return 1
  fi
}

function sleep_pod() {
  echo "Job complete. Requesting stop for $RUNPOD_POD_ID key=$RUNPOD_API_KEY external=$RUNPOD_API_KEY_EXTERNAL"

  echo "Authorization: Bearer $RUNPOD_API_KEY_EXTERNAL"
  echo "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop"

  curl -m 2 --connect-timeout 2 -fS \
    -H "Authorization: Bearer $RUNPOD_API_KEY_EXTERNAL" \
    -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
    -w '\nCURL exit=%{exitcode} HTTP=%{http_code} bytes=%{size_download} time=%{time_total}s\n' \
    -o /tmp/stop.body || true

  echo "(If you need confirmation, check from your laptop/CI.)"

  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY_EXTERNAL" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" 
  echo "Pod stopped."
}