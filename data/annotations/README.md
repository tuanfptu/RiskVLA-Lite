# Nexar human action annotations

The schema in `nexar_action_schema.json` is the only active action contract.

Do not commit generated action labels. Candidate lists and completed JSONL
files are local annotation products. An empty directory is the expected state
until a person labels clips.

Official Nexar metadata stays separate from the action. Copying
`collision_label`, `time_of_event`, or `time_of_alert` into an annotation
records the source clip. It does not decide `action` or `actionable_from`.
