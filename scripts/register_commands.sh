# Configuration for interactive shell working directories
JOZO_WORKDIR="/c/p/assignment1-basics"
POD_WORKDIR="/workspace"
JOZODOC_CONTAINER="assignment1-basics_devcontainer-cs336-dev-1"

function load_runpod_env() {
  # Check if all required variables are already set
  if [[ -n "$RP_IP" && -n "$RP_SSH_PORT" && -n "$POD_ID" ]]; then
    # Variables already loaded, just export them to ensure they're available
    export RP_IP RP_SSH_PORT POD_ID
    echo "Reusing POD_ID=${POD_ID} RP_IP=${RP_IP} RP_SSH_PORT=${RP_SSH_PORT}"
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
    export RP_IP RP_SSH_PORT POD_IDt 
    return 0
  else
    echo "Error: Failed to load runpod environment" >&2
    echo "Output from runpod_environment.sh:" >&2
    echo "$all_output" >&2
    return 1
  fi
}

function encode_args() {
  local encoded
  printf -v encoded '%q ' "$@"
  printf '%s' "$encoded" | base64 -w 0
}

function encode_command() {
  echo "eval \$(printf \\\\042)\$(base64 -d <<< $(encode_args "$@"))\$(printf \\\\042)"
}


# Generic function for running commands on remote targets
# Environment variables: RUN_SETUP_CMD, RUN_SSH_CMD, RUN_SHELL_STARTUP, RUN_WORKDIR
# Use \${INTERACTIVE} in commands to add -t flag only for interactive mode
function run_generic() {
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
    ${ssh_cmd} "$shell_startup -c \"cd $RUN_WORKDIR && export && . ./scripts/register_commands.sh && ${INIT} && exec bash --rcfile <(echo 'source ./scripts/register_commands.sh; source ~/.bashrc')\""
  else
    local ssh_cmd=$(eval echo "$RUN_SSH_CMD")
    local shell_startup=$(eval echo "$RUN_SHELL_STARTUP")

    ${ssh_cmd} "$shell_startup -c \"cd $RUN_WORKDIR && export && . ./scripts/register_commands.sh && ${INIT} && $(encode_command "$@")\""
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

export -f sleep_pod