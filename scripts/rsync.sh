function rsync_default() {
  rsync -avhz --progress --no-owner --no-group \
    ${RSYNC_EXTRA_FLAGS} \
    --exclude '.git/' \
    --exclude 'experiments/' \
    --exclude '.devcontainer/.uv-cache/' \
    --exclude '.devcontainer/.venv/' \
    --exclude '.venv/' \
    --exclude 'wandb/' \
    --exclude '__pycache__/' \
    --exclude '.uvcache/' \
    --exclude '.pytest_cache/' \
    --exclude 'uv.lock' \
    --exclude '*.egg-info/' \
    --exclude 'trace*.json' \
    "$@" 
}