# Configuration for interactive shell working directories
JOZO_WORKDIR="/c/p/assignment1-basics"
POD_WORKDIR="/workspace"
JOZODOC_CONTAINER="assignment1-basics_devcontainer-cs336-dev-1"

# Debug control - set to 1 to enable debug output, 0 to disable
DEBUG_SCRIPT=${DEBUG_SCRIPT:-1}

function load_runpod_env() {
  # Check if all required variables are already set
  if [[ -n "$RP_IP" && -n "$RP_SSH_PORT" && -n "$POD_ID" ]]; then
    # Variables already loaded, just export them to ensure they're available
    export RP_IP RP_SSH_PORT POD_ID
    # echo "Reusing POD_ID=${POD_ID} RP_IP=${RP_IP} RP_SSH_PORT=${RP_SSH_PORT}"
    return 0
  fi

  # Safely load runpod environment variables without crashing the shell
  local env_vars
  local all_output
  if all_output=$(bash -c '. /workspace/scripts/runpod_environment.sh && echo "RP_IP=$RP_IP" && echo "RP_SSH_PORT=$RP_SSH_PORT" && echo "POD_ID=$POD_ID"' 2>&1); then
    # Extract only the environment variable lines for eval
    env_vars=$(echo "$all_output" | grep -E "^(RP_IP|RP_SSH_PORT|POD_ID)=")
    eval "$env_vars"
    # Export the variables so they're cached and available to child processes
    export RP_IP RP_SSH_PORT POD_ID
    return 0
  else
    echo "Error: Failed to load runpod environment" >&2
    echo "Output from runpod_environment.sh:" >&2
    echo "$all_output" >&2
    return 1
  fi
}

function debug_args() {
  if [[ "$DEBUG_SCRIPT" -ge 1 ]]; then
    echo "Number of arguments: $#" >&2
    echo "All args as one string (\$*): '$*'" >&2
    echo "All args as separate (\$@): '$@'" >&2
    printf '%s' "$*" >&2
    for i in $(seq 1 $#); do
      echo "Arg $i: '${!i}'" >&2
    done
  fi
}

function encode_args() {
  printf '%s' "$*" | base64 -w 0
}

function encode_command() {
  if [[ "$DEBUG_SCRIPT" -ge 2 ]]; then
    echo "=== ENCODE_COMMAND DEBUG ===" >&2
    debug_args "$@"
    echo "============================" >&2
  fi
  local encoded_cmd="eval \$(printf \\\\042)\$(base64 -d <<< $(encode_args "$@"))\$(printf \\\\042)"
  if [[ "$DEBUG_SCRIPT" -ge 2 ]]; then
    echo "Final encoded command: $encoded_cmd" >&2
  fi
  echo "$encoded_cmd"
}


# Generic function for running commands on remote targets
# Environment variables: RUN_SETUP_CMD, RUN_SSH_CMD, RUN_SHELL_STARTUP, RUN_WORKDIR
# Use \${INTERACTIVE} in commands to add -t flag only for interactive mode
function run_generic() {
  if [[ "$DEBUG_SCRIPT" -ge 1 ]]; then
    echo "=== RUN_GENERIC DEBUG ==="
    debug_args "$@"
    echo "========================="
  fi
  
  # Execute setup command if provided (e.g., load_runpod_env)
  if [ -n "$RUN_SETUP_CMD" ]; then
    if ! eval "$RUN_SETUP_CMD"; then
      return 1
    fi
  fi
  
  if [ $# -eq 0 ]; then
    # No arguments - start interactive shell (set INTERACTIVE flag)
    local ssh_cmd=$(eval echo "$RUN_SSH_CMD")
    local shell_startup=$(eval echo "$RUN_SHELL_STARTUP")

    # No arguments - start interactive shell
    ${ssh_cmd} "$shell_startup -c \"cd $RUN_WORKDIR && export DEBUG_SCRIPT=$DEBUG_SCRIPT && . ./scripts/register_commands.sh && ${INIT} && exec bash --rcfile <(echo 'source ./scripts/register_commands.sh; source ~/.bashrc')\""
  else
    local ssh_cmd=$(eval echo "$RUN_SSH_CMD")
    local shell_startup=$(eval echo "$RUN_SHELL_STARTUP")
    local full_command="$shell_startup -c \"cd $RUN_WORKDIR && export DEBUG_SCRIPT=$DEBUG_SCRIPT && . ./scripts/register_commands.sh && ${INIT} && $(encode_command "$@")\""
    
    if [[ "$DEBUG_SCRIPT" -ge 1 ]]; then
      echo "SSH command: $ssh_cmd"
      echo "Full remote command: $full_command"
    fi
    
    ${ssh_cmd} "$full_command"
  fi
}

function run_jozo() {
  RUN_SETUP_CMD="" \
  RUN_SSH_CMD="ssh -t kruno@jozo -i ~/.ssh/id_jozo" \
  RUN_SHELL_STARTUP="bash" \
  RUN_WORKDIR="/mnt/${JOZO_WORKDIR}" \
  INIT="true" \
  run_generic "$@"
}

function run_jozodoc() {
  RUN_SETUP_CMD="" \
  RUN_SSH_CMD="ssh -t kruno@jozo -i ~/.ssh/id_jozo" \
  RUN_SHELL_STARTUP="docker exec -it ${JOZODOC_CONTAINER} bash" \
  RUN_WORKDIR="/workspace" \
  INIT="true" \
  run_generic "$@"
}

function run_pod() {
  RUN_SETUP_CMD="load_runpod_env" \
  RUN_SSH_CMD="ssh -t -i ~/.ssh/id_runpod -p \$RP_SSH_PORT root@\$RP_IP" \
  RUN_SHELL_STARTUP="bash" \
  RUN_WORKDIR="${POD_WORKDIR}" \
  INIT="export \$(cat /proc/1/environ | tr '\\000' '\\n' | grep -E '^(JUPYTER_PASSWORD|WANDB_API_KEY|RUNPOD_API_KEY_EXTERNAL|RUNPOD_POD_ID|RUNPOD_GPU_COUNT|RUNPOD_MEM_GB)=' | xargs)" \
  run_generic "$@"
}

run_podscreen() {
  # echo screen -S background bash -c "'$(encode_command "$*")'"
  # run_pod screen -S background bash -c "$*"
  run_pod screen -S background bash -c "echo stay && sleep 120"
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
function push_jozodoc() {
  push_jozo
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
function push_podscreen() {
  push_pod
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

function pushrun() {
  local target="$1"
  shift
  
  if [[ "$DEBUG_SCRIPT" -ge 1 ]]; then
    echo "=== PUSHRUN DEBUG ==="
    echo "Target: $target"
    debug_args "$@"
    echo "===================="
  fi
  
  echo "Push and run on ${target}..."
  if push "$target"; then
    run "$target" "$@"
  else
    echo "Push failed, aborting run"
    return 1
  fi
}

function sleep_pod() {
  if ! [[ -n "${RUNPOD_POD_ID}" ]]; then
    run pod sleep_pod
    return 0
  fi
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

function sleep_jozo() {
  if ! [[ -f "/mnt/c/Users/kruno/sleep_computer.bat" ]]; then
    run jozo sleep_jozo
    return 0
  fi
  run_jozo /mnt/c/Users/kruno/sleep_computer.bat
}

export -f sleep_pod