# Auto-fetch POD_ID if not specified
if [ -z "$POD_ID" ]; then
  # echo "POD_ID not specified, fetching available pods..."
  POD_ID=$(curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods" \
    | jq -r '.[] | .id' \
    | head -n 1)
  
  if [ -z "$POD_ID" ]; then
    echo "Error: No running pods found"
    exit 1
  else
    echo "Auto-selected POD_ID: $POD_ID"
  fi
fi

export POD_ID

read RP_IP RP_SSH_PORT < <(
  curl -sS -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://rest.runpod.io/v1/pods/$POD_ID" \
  | jq -r '[.publicIp, .portMappings["22"]] | @tsv'
)

export RP_IP RP_SSH_PORT

# Output variables in the format expected by load_runpod_env
echo "RP_IP=$RP_IP"
echo "RP_SSH_PORT=$RP_SSH_PORT"
echo "POD_ID=$POD_ID"