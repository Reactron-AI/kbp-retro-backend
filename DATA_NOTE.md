# Excluded from this repo

This repo has source code only. Model checkpoints and large datasets are excluded (kept on the lab machine):

- `retro_star/retro_star/dataset/origin_dict.csv` (~1.3G) — building-block stock list
- `retro_star/retro_star/saved_models/best_epoch_final_4.pt` — retro* value function checkpoint
- `retro_star/retro_star/one_step_model/` — the `mlp` planner's one-step model (unused by the default `localretro` config)
- `LocalRetro/models/LocalRetro_USPTO_50K.pth` (~114M) — LocalRetro checkpoint for the default `localretro` planner
- `LocalRetro/models/LocalRetro_Pistachio_epoch_45.pth` (~114M) — LocalRetro checkpoint for the `interretro` planner's one-step model
- `InterRetro/InterRetro/experiments/default0429-075316-s51/models_epoch4_it0.pth` (~82M) — InterRetro checkpoint
- `backend/output*/`, `backend/data/`, `backend/logs/`, `backend/*.log` — runtime-generated route artifacts, the live feedback/saved-pathway store, and logs (not source, and `backend/data/*.json` is real collaborator-submitted data, not something to publish)

Default planner is `localretro` (`RSPlannerLocalRetro` in `retro_star/retro_star/api.py`). `interretro` is also a fully supported planner option (see `backend/retro_wrapper/planner.py`) — it just needs `torch==2.2.1` in its own conda env, run as a separate process from the `localretro`/`mlp` instance (which is pinned to `torch==1.9.0`); it is **not** on hold, despite what an older version of this note said.

See `backend/.env.example` for the runtime config (planner choice, GPU, paths, CORS, API key).
