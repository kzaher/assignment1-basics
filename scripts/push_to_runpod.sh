export POD_ID=vi855obf5a5lx2
read RP_IP RP_SSH_PORT < <(
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$POD_ID" \
  | jq -r '[.publicIp, .portMappings["22"]] | @tsv'
)

echo "IP:   $RP_IP"
echo "Port: $RP_SSH_PORT"

rsync -avhL --progress \
  --exclude '.git/' \
  --exclude 'experiments/' \
  --exclude '.devcontainer/.uv-cache/' \
  --exclude '.devcontainer/.venv/' \
  --exclude '.venv/' \
  --exclude 'wandb/' \
  -e "ssh -p $RP_SSH_PORT" \
  . root@"$RP_IP":/workspace