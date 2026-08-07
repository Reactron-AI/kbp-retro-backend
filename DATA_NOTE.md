# Excluded from this repo

This repo has source code only. Model checkpoints and large datasets are excluded (kept on the lab devbox):

- `retro_star/retro_star/dataset/origin_dict.csv` (1.3G) — building-block stock list
- `retro_star/retro_star/saved_models/best_epoch_final_4.pt` (1.1M) — retro* value function checkpoint
- `LocalRetro/models/LocalRetro_Pistachio_epoch_45.pth` (~114M) — LocalRetro one-step model checkpoint

Default planner is `localretro` (`RSPlannerLocalRetro` in `retro_star/retro_star/api.py`). `retro_star/retro_star/one_step_model/` and `InterRetro/` (the `mlp`/`interretro` planner paths) are not included since they're unused by the default config.

See `backend/.env.example` for the runtime config (planner choice, paths, CORS, API key).
