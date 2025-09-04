export POD_ID=${POD_ID:-6wwvsj886w5wx3}

read RP_IP RP_SSH_PORT < <(
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$POD_ID" \
  | jq -r '[.publicIp, .portMappings["22"]] | @tsv'
)

echo "IP:   $RP_IP"
echo "Port: $RP_SSH_PORT"