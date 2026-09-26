# Model artifacts

No pretrained model is bundled. Run `training.launch.py` with a writable
`model_directory` to generate `best.pt` and `best.metrics.json` after a completed
evaluation improves mean coverage. See the package README for commands.

`best.pt` is the package's versioned Python-pickle checkpoint despite its suffix.
Load only files you trust. It stores online and target network weights,
dimensions, training configuration, and evaluation metrics. It is not a complete
training-resumption snapshot. The adjacent JSON is human-readable evaluation
evidence; a smoke checkpoint proves the pipeline executes, not that the policy
has learned effective exploration.

Keep experiment output outside this installed share directory, for example
`/ros2_ws/models/experiment-001`, and pass its checkpoint's absolute path to
`mission.launch.py model_path:=...`.
