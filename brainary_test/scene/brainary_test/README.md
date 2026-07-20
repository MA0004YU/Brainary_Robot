# v1_active/

The ONE active V1 scene. Written by `layout_editor/layout_v1_ui.py` "Save V1":
- `scene_v1_latest.usd`     — active scene snapshot
- `scene_v1_latest.json`    — manifest (objects + xforms)
- `scene_v1_<ts>.usd|json`  — timestamped backups
- `scene_v1_registry.json`  — the pointer all consumers read (committed seed here)

On a fresh checkout only `scene_v1_registry.json` exists; consumers fall back to
the base task cfg until you save a scene.
