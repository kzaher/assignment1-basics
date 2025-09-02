# runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

bash -lc '
set -euo pipefail

apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server rsync screen vim

# 2) Prepare runtime dirs & host keys
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
# Generate host keys if missing
ssh-keygen -A

# 4) Authorized keys
echo "$SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# 6) Launch sshd in foreground (no systemd)

# exec /usr/sbin/sshd -D -e

# 5) Minimal sshd config hardening (key-only root login)
sed -ri "s|^#?PasswordAuthentication .*|PasswordAuthentication no|" /etc/ssh/sshd_config
sed -ri "s|^#?PermitRootLogin .*|PermitRootLogin prohibit-password|" /etc/ssh/sshd_config
# Ensure default AuthorizedKeysFile is used (usually already is):
# sed -ri "s|^#?AuthorizedKeysFile .*|AuthorizedKeysFile .ssh/authorized_keys|" /etc/ssh/sshd_config

# (Optional) Listen on all if needed:
# if ! grep -q "^ListenAddress" /etc/ssh/sshd_config; then
#   echo "ListenAddress 0.0.0.0" >> /etc/ssh/sshd_config
# fi

pip install uv

cd /workspace

echo "Running my long job"
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

echo "Sleeping, waiting stop and waiting for ssh connection"
exec /usr/sbin/sshd -D -e
'