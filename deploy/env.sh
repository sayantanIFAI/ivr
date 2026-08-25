# Environment for every service in the stack. Source before launching.
#
# Everything referenced here MUST live under /workspace. RunPod wipes the
# container's overlay filesystem ("/" and "/root") on every restart and
# keeps only the network volume, so anything installed to /usr/local/bin
# or stored in /var/lib is gone the next time the pod boots. That is not a
# hypothetical: the ollama binary and the entire Postgres installation
# have each been lost to it three times.

export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export HF_HUB_ENABLE_HF_TRANSFER=0

export OLLAMA_MODELS=/workspace/.ollama/models
# -1 keeps models resident indefinitely. Without it Ollama unloads after
# five idle minutes and the next caller pays a 47s cold start mid-call.
export OLLAMA_KEEP_ALIVE=-1

export PYTHONPATH=/workspace/AI4Bharat_NeMo:/workspace/kolkata-care-voice-agent:

# DATABASE_URL is deliberately NOT set: clinic-api/db.py then defaults to
# sqlite:////workspace/clinic.db, which persists across restarts. Postgres
# physically cannot run on this volume -- see that file's docstring. Set
# this only when pointing at a real external Postgres.
export CLINIC_API_BASE=http://localhost:8080
export TTS_URL=http://localhost:8002/synthesize
export SILERO_VAD_REPO=/workspace/silero-vad

# /workspace/bin first: that is where the persistent ollama binary lives.
export PATH=/workspace/bin:/workspace/venv/bin:${PATH:-}
