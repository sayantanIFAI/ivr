#!/bin/bash
# ============================================================================
# Kolkata Care Diagnostics voice agent -- ADD-ON pod setup.
#
# Run this AFTER voice-to-rx-repo/setup_pod.sh, on the SAME pod. This does
# NOT reinstall NeMo, Ollama, or Silero VAD -- they're already provisioned
# and this service imports voicerx.asr directly to reuse them. Re-running
# the full setup_pod.sh here would waste ~8 minutes for nothing new.
#
# Run:  export HF_TOKEN=hf_xxx   (only needed if voicerx's install skipped it)
#       bash setup_addon.sh 2>&1 | tee /workspace/setup_addon.log
# ============================================================================
set -euo pipefail

VOICERX_DIR="${VOICERX_DIR:-/workspace/voice-to-rx-repo}"
AGENT_DIR="${AGENT_DIR:-/workspace/kolkata-care-voice-agent}"

echo "=============================================================="
echo " 0. Preflight -- confirm the shared stack is actually there"
echo "=============================================================="
[ -d "$VOICERX_DIR" ] || {
    echo "FATAL: $VOICERX_DIR not found. This service reuses its ASR"
    echo "integration on purpose -- clone/deploy that repo first."; exit 1; }

python3 -c "import nemo.collections.asr" 2>/dev/null || {
    echo "FATAL: nemo not importable. This is the SAME known issue noted in"
    echo "voice-to-rx-repo/HANDOFF.md ('NeMo ASR Not Loading'). Fix that"
    echo "first -- this new service depends on the exact same import path,"
    echo "not a separate one:"
    echo '  export PYTHONPATH=/workspace/AI4Bharat_NeMo:/workspace/pylibs:'"$VOICERX_DIR"':$PYTHONPATH'
    exit 1
}
echo "  nemo import OK"

command -v ollama >/dev/null || { echo "FATAL: ollama not found -- run voice-to-rx-repo's setup first."; exit 1; }
(ollama list | grep -q "qwen2.5:7b") || { echo "FATAL: qwen2.5:7b not pulled -- run voice-to-rx-repo's setup first."; exit 1; }
echo "  ollama + qwen2.5:7b OK"

echo "=============================================================="
echo " 1. This service's own dependencies"
echo "=============================================================="
pip install --quiet httpx

echo "=============================================================="
echo " 2. AI4Bharat Indic-TTS (the one genuinely new model)"
echo "=============================================================="
# github.com/AI4Bharat/Indic-TTS -- FastPitch + HiFi-GAN-V1, Bengali
# checkpoint hosted on Bhashini. UNLIKE IndicConformer/Ollama, nothing in
# this account has deployed this before -- budget time to work through its
# own README, and test /synthesize manually before wiring the agent to it.
if [ ! -d /workspace/Indic-TTS ]; then
    git clone --depth 1 https://github.com/AI4Bharat/Indic-TTS.git /workspace/Indic-TTS
fi
echo "  Indic-TTS cloned to /workspace/Indic-TTS -- follow ITS README to"
echo "  download the Bengali FastPitch+HiFi-GAN checkpoint and start its"
echo "  server. Point this service at it via:"
echo "    export TTS_URL=http://localhost:8001/synthesize"

echo "=============================================================="
echo " 3. Record the fallback audio set (do this once, by hand)"
echo "=============================================================="
mkdir -p "$AGENT_DIR/static/fallback_audio"
cat <<'EOF'
  These 3 files must exist before going live -- they are what plays when
  live TTS itself is down, so they must NOT depend on the TTS service:

    static/fallback_audio/sorry_repeat.wav   "দুঃখিত, শুনতে পাইনি, আবার বলুন"
    static/fallback_audio/system_busy.wav    "একটু সমস্যা হচ্ছে, একটু ধরুন"
    static/fallback_audio/check_failed.wav   "এখনই দেখতে পারছি না, স্টাফের কাছে দিচ্ছি"

  Easiest path: synthesize each ONCE with Indic-TTS once it's working,
  save the output here, and never regenerate them live. A human recording
  is even more robust and costs about ten minutes.
EOF

echo "=============================================================="
echo " DONE"
echo "=============================================================="
echo "Start with:"
echo "  cd $AGENT_DIR"
echo "  export PYTHONPATH=/workspace/AI4Bharat_NeMo:$VOICERX_DIR:\$PYTHONPATH"
echo "  export CLINIC_API_BASE=http://localhost:8080   # the Spring Boot service"
echo "  export TTS_URL=http://localhost:8001/synthesize"
echo "  python3 -m uvicorn main:app --host 0.0.0.0 --port 8100"
echo
echo "Expose port 8100 in the RunPod console (VoiceToRx already owns 8000"
echo "on this pod) -- the proxy URL will be https://<pod-id>-8100.proxy.runpod.net"
