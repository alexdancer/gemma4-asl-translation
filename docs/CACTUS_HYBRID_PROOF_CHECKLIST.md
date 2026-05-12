# Cactus Hybrid Proof Checklist (#75)

Use this to verify the Arch-hosted Cactus hybrid service is correctly wired and fail-closed.

## Prereqs

- Service installed as `asl-cactus-hybrid.service`
- Env file exists at `/etc/asl/cactus-hybrid.env`
- `jq` installed

Required env entries in `/etc/asl/cactus-hybrid.env`:

- `ASL_CACTUS_SERVICE_API_KEY` = local shared bearer secret (caller -> hybrid service)
- `ASL_HF_OPENAI_BASE_URL` = HF OpenAI-compatible base URL
- `ASL_HF_TOKEN` = Hugging Face token for cloud handoff
- `ASL_CACTUS_MODEL_VERSION` = model version label for proof response

## One-command proof run

From repo root:

```bash
bash scripts/runtime/prove_cactus_hybrid.sh
```

Optional overrides:

```bash
SERVICE_URL=http://127.0.0.1:9000/ ENV_FILE=/etc/asl/cactus-hybrid.env bash scripts/runtime/prove_cactus_hybrid.sh
```

## Expected result

The script should print:

1. Success response JSON containing required proof fields:
   - `runtime_mode`
   - `cloud_handoff`
   - `model_id`
   - `model_version`
2. A fail-closed response with:
   - `error_code: CLOUD_HANDOFF_FAILED`
3. Final restoration message confirming the original HF token was restored and service restarted.

## Notes

- The script temporarily breaks `ASL_HF_TOKEN` to verify fail-closed behavior, then restores the original env automatically.
- If the script exits early, re-run:

```bash
sudo systemctl restart asl-cactus-hybrid.service
```
