#!/usr/bin/env bash
# =============================================================================
# GCP Setup Script — Run this in Google Cloud Shell
# (Click the >_ icon in the top-right of console.cloud.google.com)
#
# This script creates an L4 GPU VM for Simultaneous MT training & inference.
# Budget: ~₹9,000 for 4 days continuous (~$1.10/hr total).
#
# Prerequisites:
#   1. A Google Cloud account with billing enabled
#   2. GCP credits linked to the billing account
#
# Usage:
#   bash gcp_setup.sh
# =============================================================================
set -euo pipefail

# ---------- Configuration (edit if needed) ----------
PROJECT_ID="${GCP_PROJECT_ID:-simul-mt-project}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="simul-mt-vm"
MACHINE_TYPE="g2-standard-8"       # 8 vCPU, 32 GB RAM, 1x L4 GPU
GPU_TYPE="nvidia-l4"
GPU_COUNT=1
DISK_SIZE="100GB"
DISK_TYPE="pd-ssd"
# Deep Learning VM image with CUDA 12.4 + PyTorch pre-installed
IMAGE_FAMILY="common-cu124-debian-11"
IMAGE_PROJECT="deeplearning-platform-release"
# -----------------------------------------------------

echo "=============================================="
echo "  Simultaneous MT — GCP VM Setup"
echo "=============================================="
echo ""
echo "Project:  ${PROJECT_ID}"
echo "Zone:     ${ZONE}"
echo "Machine:  ${MACHINE_TYPE} + ${GPU_COUNT}x ${GPU_TYPE}"
echo "Disk:     ${DISK_SIZE} ${DISK_TYPE}"
echo ""

# --- Step 1: Set project ---
echo "[1/5] Setting active project..."
gcloud config set project "${PROJECT_ID}" 2>/dev/null || {
    echo "Project '${PROJECT_ID}' not found. Creating it..."
    gcloud projects create "${PROJECT_ID}" --name="Simultaneous MT"
    gcloud config set project "${PROJECT_ID}"
    echo ""
    echo ">>> IMPORTANT: You must link a billing account to this project!"
    echo "    Go to: https://console.cloud.google.com/billing/linkedaccount?project=${PROJECT_ID}"
    echo "    Select your billing account with the GCP credits."
    echo "    Then re-run this script."
    echo ""
    read -rp "Press Enter after you've linked billing (or Ctrl-C to exit)..."
}

# --- Step 2: Enable required APIs ---
echo "[2/5] Enabling Compute Engine API..."
gcloud services enable compute.googleapis.com

# --- Step 3: Check / request GPU quota ---
echo "[3/5] Checking GPU quota in ${ZONE}..."
REGION="${ZONE%-*}"
QUOTA=$(gcloud compute regions describe "${REGION}" \
    --format="value(quotas[metric=NVIDIA_L4_GPUS].limit)" 2>/dev/null || echo "0")
if [ "${QUOTA}" = "0" ] || [ -z "${QUOTA}" ]; then
    echo ""
    echo ">>> GPU quota for NVIDIA_L4_GPUS in ${REGION} is 0."
    echo "    You need to request a quota increase:"
    echo "    1. Go to: https://console.cloud.google.com/iam-admin/quotas"
    echo "    2. Filter: 'NVIDIA L4' and region '${REGION}'"
    echo "    3. Select the quota → Edit Quotas → Request limit: 1"
    echo "    4. Fill in your details → Submit"
    echo "    (Usually approved within 5-15 minutes)"
    echo ""
    read -rp "Press Enter after your quota is approved (or Ctrl-C to exit)..."
fi

# --- Step 4: Create firewall rule for frontend (port 8080) ---
echo "[4/5] Creating firewall rule for frontend access..."
gcloud compute firewall-rules create allow-frontend-8080 \
    --allow=tcp:8080 \
    --target-tags=http-server \
    --description="Allow access to SiMT frontend on port 8080" \
    --direction=INGRESS \
    2>/dev/null || echo "  (firewall rule already exists)"

# --- Step 5: Create the VM ---
echo "[5/5] Creating VM instance '${INSTANCE_NAME}'..."
gcloud compute instances create "${INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --image-family="${IMAGE_FAMILY}" \
    --image-project="${IMAGE_PROJECT}" \
    --boot-disk-size="${DISK_SIZE}" \
    --boot-disk-type="${DISK_TYPE}" \
    --maintenance-policy=TERMINATE \
    --tags=http-server \
    --metadata=install-nvidia-driver=True \
    --scopes=default,storage-rw

echo ""
echo "=============================================="
echo "  VM Created Successfully!"
echo "=============================================="
echo ""
echo "To connect via SSH:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
echo ""
echo "Once connected, run:"
echo "  git clone https://github.com/chiranjeevi-sagi/Simultaneous-Machine-Translation.git"
echo "  cd Simultaneous-Machine-Translation"
echo "  bash gcp_vm_setup.sh"
echo ""
echo "To check VM external IP (for frontend access):"
echo "  gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} \\"
echo "    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'"
echo ""
echo "To STOP the VM (saves money when not in use):"
echo "  gcloud compute instances stop ${INSTANCE_NAME} --zone=${ZONE}"
echo ""
echo "To START it again:"
echo "  gcloud compute instances start ${INSTANCE_NAME} --zone=${ZONE}"
echo ""
echo "To DELETE the VM when done (after June 27):"
echo "  gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE}"
