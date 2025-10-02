# Configuration for interactive shell working directories
JOZO_WORKDIR="/c/p/assignment1-basics"
POD_WORKDIR="/workspace"
JOZODOC_CONTAINER="assignment1-basics_devcontainer-cs336-dev-1"

# Debug control - set to 1 to enable debug output, 0 to disable
DEBUG_SCRIPT=${DEBUG_SCRIPT:-0}

function load_runpod_env() {
  # Check if all required variables are already set
  if [[ -n "$RP_IP" && -n "$RP_SSH_PORT" && -n "$POD_ID" && "$RP_SSH_PORT_POD_ID" == "$POD_ID" ]]; then
    # Variables already loaded, just export them to ensure they're available
    export RP_IP RP_SSH_PORT POD_ID
    if [[ ${DEBUG_SCRIPT} -ge 1 ]]; then
      echo "Reusing POD_ID=${POD_ID} RP_IP=${RP_IP} RP_SSH_PORT=${RP_SSH_PORT}"
    fi
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
    export RP_SSH_PORT_POD_ID="${POD_ID}"
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
    for i in $(seq 1 $#); do
      echo "Arg $i: '${!i}'" >&2
    done
  fi
}

function encode_args() {
  printf '%s\0' "$@" | base64 -w 0
}

function exec_base64_command() {
  mapfile -d '' -t argv < <(echo "${1}" | base64 -d );
  if [[ "$DEBUG_SCRIPT" -ge 2 ]]; then
    echo "=== exec_base64_command DEBUG ===" >&2
    debug_args "${argv[@]}"
    echo "============================" >&2
  fi
  if [[ ${#argv[@]} -gt 0 && "${argv[0]}" != "" ]]; then
    exec "${argv[@]}"
  fi
}

function eval_base64_command() {
  mapfile -d '' -t argv < <(echo "${1}" | base64 -d ); 
  if [[ "$DEBUG_SCRIPT" -ge 2 ]]; then
    echo "=== eval_base64_command DEBUG ===" >&2
    debug_args "${argv[@]}"
    echo "============================" >&2
  fi
  eval "${argv[@]}"
}

function initialize_run() {
  # Forwards process environment variables.
  export $(cat /proc/1/environ | tr '\000' '\n' | grep -E '^(JUPYTER_PASSWORD|WANDB_API_KEY|RUNPOD_API_KEY_EXTERNAL|RUNPOD_POD_ID|RUNPOD_GPU_COUNT|RUNPOD_MEM_GB)=' | xargs)
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

  TMUX_STARTED=0;
  if [[ -n "$@" ]]; then
    TMUX_STARTED=1;
  fi
  INTERPRETER=${INTERPRETER:-exec_base64_command}
  $(eval echo "${RUN_SSH_CMD}") "$RUN_SHELL_STARTUP \"cd $RUN_WORKDIR && export DEBUG_SCRIPT=$DEBUG_SCRIPT && exec bash --rcfile <(echo 'source ./scripts/register_commands.sh; TMUX_STARTED=${TMUX_STARTED} source ~/.bashrc; initialize_run; ${INTERPRETER} $(encode_args "$@")');\""
}

function run_jozo() {
  RUN_SETUP_CMD="" \
  RUN_SSH_CMD="ssh -t kruno@${SSH_HOSTNAME:-jozo.tailb3978.ts.net} -i ~/.ssh/id_jozo" \
  RUN_SHELL_STARTUP="bash -c" \
  RUN_WORKDIR="/mnt/${JOZO_WORKDIR}" \
  run_generic "$@"
}

function run_jozodoc() {
  RUN_SETUP_CMD="" \
  RUN_SSH_CMD="ssh -t kruno@${SSH_HOSTNAME:-jozo.tailb3978.ts.net} -i ~/.ssh/id_jozo" \
  RUN_SHELL_STARTUP="docker exec -it ${JOZODOC_CONTAINER} bash -c " \
  RUN_WORKDIR="/workspace" \
  run_generic "$@"
}

function run_pod() {
  RUN_SETUP_CMD="load_runpod_env" \
  RUN_SSH_CMD="ssh -t -i ~/.ssh/id_runpod -p \${RP_SSH_PORT} root@\${RP_IP}" \
  RUN_SHELL_STARTUP="${RUN_SHELL_STARTUP:-bash -c}" \
  RUN_WORKDIR="${POD_WORKDIR}" \
  run_generic "$@"
}

function load_vast_ai() {
  export VASTAI_ID=${VASTAI_ID:-$(vastai show instances -q | head -n1)}
  if [[ "$VASTAI_CACHED_ID" == "$VASTAI_ID" ]]; then
    return 0
  fi
  export VASTAI_SSH="$(vastai ssh-url $VASTAI_ID)"
  export VASTAI_CACHED_ID="$VASTAI_ID"
}

function run_vast() {
  load_vast_ai
  RUN_SETUP_CMD="" \
  RUN_SSH_CMD="ssh -t -i ~/.ssh/id_runpod `vastai ssh-url ${VASTAI_ID}`" \
  RUN_SHELL_STARTUP="${RUN_SHELL_STARTUP:-bash -c}" \
  RUN_WORKDIR="${POD_WORKDIR}" \
  run_generic "$@"
}

run_podscreen() {
  if [[ "$DEBUG_SCRIPT" -ge 1 ]]; then
    echo "=== RUN_PODSCREEN DEBUG ==="
    debug_args "$@"
    echo "========================="
  fi
  RUN_SHELL_STARTUP="screen -S background bash -c" \
  run_pod "$@"
}

run_vasttmux() {
  if [[ "$DEBUG_SCRIPT" -ge 1 ]]; then
    echo "=== RUN_PODSCREEN DEBUG ==="
    debug_args "$@"
    echo "========================="
  fi
  RUN_SHELL_STARTUP="tmux new-session -s background" \
  run_vast "$@"
}

# Generic dispatcher function for commands with target-specific implementations
function dispatch_to_target() {
  local command_prefix="$1"
  local target="$2"
  shift 2
  local func_name="${command_prefix}_${target}"
  
  if declare -f "$func_name" > /dev/null; then
    "$func_name" "$@"
  else
    # Automatically detect available targets
    local available_targets=$(declare -F | grep "^declare -f ${command_prefix}_" | sed "s/declare -f ${command_prefix}_//" | tr '\n' ' ')
    echo "Error: Unknown target '$target'. Available targets: ${available_targets}"
    return 1
  fi
}

function run() {
  dispatch_to_target "run" "$@"
}

function push_jozo() {
  . ./scripts/rsync.sh
  # Use -l to preserve symbolic links for jozo
  RSYNC_EXTRA_FLAGS="" RSYNC_RSH="ssh -i ~/.ssh/id_jozo" rsync_default --exclude 'data/' --exclude='*.toml' --exclude='*.lock' --rsync-path="wsl rsync" . kruno@${SSH_HOSTNAME:-jozo.tailb3978.ts.net}:/mnt/c/p/assignment1-basics/
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

function push_vast() {
  load_vast_ai
  . ./scripts/rsync.sh
  # Use -L to follow/convert symbolic links for vastai
  PORT=$(echo ${VASTAI_SSH} | cut -d ':' -f 3)
  IP=$(echo ${VASTAI_SSH} | cut -d '@' -f 2 | cut -d ':' -f 1)
  RSYNC_EXTRA_FLAGS="-L" RSYNC_RSH="ssh -i ~/.ssh/id_runpod -p ${PORT}" rsync_default \
    --include 'data/' \
    --include 'data/owt_train.txt.tokens.vocab_size=32000.npy' \
    --include 'data/owt_valid.txt.tokens.vocab_size=32000.npy' \
    --exclude 'data/*' \
    . root@"${IP}":/workspace
}

function push_vasttmux() {
  push_vast
}

function push() {
  dispatch_to_target "push" "$@"
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

function down_pod() {
  if ! [[ -n "${RUNPOD_POD_ID}" ]]; then
    INTERPRETER=eval_base64_command pushrun pod down_pod
    return 0
  fi
  down_this_pod
}

# Alias for down_pod
function pod_down() {
  down_pod "$@"
}

function down_this_pod() {
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

function down_jozo() {
  if ! [[ -f "/mnt/c/Users/kruno/sleep_computer.bat" ]]; then
    INTERPRETER=eval_base64_command run jozo down_jozo
    return 0
  fi
  down_this_jozo
}

function down_jozot() {
  if ! [[ -f "/mnt/c/Users/kruno/sleep_computer.bat" ]]; then
    INTERPRETER=eval_base64_command run jozot down_jozo
    return 0
  fi
  down_this_jozo
}

function down_this_jozo() {
  bash -c "/mnt/c/Users/kruno/sleep_computer.bat"
}

# Alias for down_jozo
function jozo_down() {
  down_jozo "$@"
}

function down() {
  if ! [[ -n "$@" ]]; then
    down_this
    return 0
  fi
  dispatch_to_target "down" "$@"
}

function up_jozo() {
  ssh k@raspberrypi.tailb3978.ts.net bash -c "'for i in {1..20} ; do wake_jozo; done;'"
  ssh k@raspberrypi.tailb3978.ts.net ping 100.110.214.59
}

function up_pod() {
  up_this_pod
}

function read_pod_status() {
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/79rzj7t4o6onxv"  |  jq -r '.desiredStatus'
}

function up_this_pod() {
  load_runpod_env
  if ! [[ -n $"POD_ID" ]]; then
    echo No pod detected
    return 1
  fi 

  STATUS=$(read_pod_status)
  if [[ "${STATUS}" == "RUNNING" ]]; then
    echo "Pod ${POD_ID} is already running. ✅"
    return 0
  fi

  echo "Starting suspended pod $POD_ID"

  echo "Authorization: Bearer $RUNPOD_API_KEY"
  echo "https://rest.runpod.io/v1/pods/$POD_ID/start"

  curl -m 2 --connect-timeout 2 -fS \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -X POST "https://rest.runpod.io/v1/pods/$POD_ID/start" \
    -w '\nCURL exit=%{exitcode} HTTP=%{http_code} bytes=%{size_download} time=%{time_total}s\n' \
    -o /tmp/resume.body || echo Failed ❗️ && return 1

  while [[ "${STATUS}" != "RUNNING" ]]; do
    STATUS=$(read_pod_status)
    sleep 1
    echo "Status ${STATUS}"
  done
  echo "Pod resumed. ✅"
}

function up() {
  dispatch_to_target "up" "$@"
}

function down_this_vast() {
  if [[ -n "$VAST_AI_API_KEY" ]]; then
    vastai set api-key $VAST_AI_API_KEY
  fi
  echo vastai stop instance $(echo "$VAST_CONTAINERLABEL" | cut -d '.' -f 2)
}

function down_this() {
    if [[ -n "${RUNPOD_POD_ID}" ]]; then
      down_this_pod
    elif [[ -f "/mnt/c/Users/kruno/sleep_computer.bat" ]]; then 
      down_this_jozo
    elif [[ -n "${VAST_CONTAINERLABEL}" ]]; then 
      down_this_vast
    else 
      echo "This survived" >&2
      return 1
    fi
}

function reset_jozo() {
  ssh k@raspberrypi.tailb3978.ts.net ./reset_jozo.py
}

function reset() {
  dispatch_to_target "reset" "$@"
}

function pull_vast() {
  load_vast_ai
  eval $(vastai ssh-url $VASTAI_ID | sed -E 's#ssh://([^@]+)@([^:]+):([0-9]+)#USER=\1 HOST=\2 PORT=\3#') \
    && scp -i ~/.ssh/id_runpod -P $PORT $USER@$HOST:/tmp/trace.json trace.json
}

function pull() {
  dispatch_to_target "pull" "$@"
}

export -f down
export -f down_this
export -f down_this_pod
export -f down_this_jozo
export -f down_this_vast