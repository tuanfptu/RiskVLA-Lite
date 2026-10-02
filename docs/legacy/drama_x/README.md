# LEGACY / ABANDONED FOR CURRENT STUDY

DRAMA-X is not the primary RiskVLA-Lite dataset. The Honda raw archive is too
large for this study, so the active benchmark is
`nexar-ai/nexar_collision_prediction`.

Retained here:

- `DATA_ACCESS_REPORT.md`: historical public-annotation audit
- `label_inventory.json`: counts from the public DRAMA-X JSONL
- `drama_media_blocker.json`: the previous blocked-media record
- `configs/legacy/drama_x_action_mapping.yaml`: the old native-label mapping
- `src/riskvla/legacy/`: the old loader
- `scripts/legacy/inspect_drama_x.py`: the old inspection script

Do not use the DRAMA-X mapping as Nexar action ground truth. Do not treat
`DRAMA_DOWNLOAD_URL` as a required secret. No private download URL is stored
in this repository.
