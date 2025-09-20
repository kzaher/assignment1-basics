function rsync_default() {
  rsync -avhLz --progress --no-owner --no-group \
    --exclude '.git/' \
    --exclude 'experiments/' \
    --exclude '.devcontainer/.uv-cache/' \
    --exclude '.devcontainer/.venv/' \
    --exclude '.venv/' \
    --exclude 'wandb/' \
    --include 'data/' \
    --include 'data/owt_train.txt.tokens.vocab_size=32000.npy' \
    --include 'data/owt_valid.txt.tokens.vocab_size=32000.npy' \
    --exclude 'data/*' \
    $@ 
}