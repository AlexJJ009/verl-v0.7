# On-Policy WDL-SFT Visualization

This is a small `uv` environment for report figures only. It reads the staged-v1
metrics JSONL files and writes publication-ready intervention plots plus a CSV
summary.

Run from the repo root:

```bash
uv run --project analysis/on_policy_wdl_viz \
  python analysis/on_policy_wdl_viz/plot_stage_intervention.py
```

Generate the boxed-prompt rerun figure:

```bash
uv run --project analysis/on_policy_wdl_viz \
  python analysis/on_policy_wdl_viz/plot_stage_intervention.py \
  --preset boxed \
  --output-stem analysis/on_policy_wdl_viz/outputs/stage_intervention_boxed_math500 \
  --summary-csv analysis/on_policy_wdl_viz/outputs/stage_intervention_boxed_summary.csv
```

Generate the plateau-handoff P60 figure:

```bash
uv run --project analysis/on_policy_wdl_viz \
  python analysis/on_policy_wdl_viz/plot_stage_intervention.py \
  --preset plateau_p60 \
  --output-stem analysis/on_policy_wdl_viz/outputs/stage_intervention_plateau_p60_math500 \
  --summary-csv analysis/on_policy_wdl_viz/outputs/stage_intervention_plateau_p60_summary.csv
```

Default outputs:

- `analysis/on_policy_wdl_viz/outputs/stage_intervention_summary.csv`
- `analysis/on_policy_wdl_viz/outputs/stage_intervention_math500.png`
- `analysis/on_policy_wdl_viz/outputs/stage_intervention_math500.pdf`
- `docs/joint_training/courses/on-policy-wdl-overleaf/images/stage_intervention_math500.png`
- `docs/joint_training/courses/on-policy-wdl-overleaf/images/stage_intervention_math500.pdf`

Boxed-prompt outputs:

- `analysis/on_policy_wdl_viz/outputs/stage_intervention_boxed_summary.csv`
- `analysis/on_policy_wdl_viz/outputs/stage_intervention_boxed_math500.png`
- `analysis/on_policy_wdl_viz/outputs/stage_intervention_boxed_math500.pdf`
- `docs/joint_training/courses/on-policy-wdl-overleaf/images/stage_intervention_boxed_math500.png`
- `docs/joint_training/courses/on-policy-wdl-overleaf/images/stage_intervention_boxed_math500.pdf`

Plateau-handoff P60 outputs:

- `analysis/on_policy_wdl_viz/outputs/stage_intervention_plateau_p60_summary.csv`
- `analysis/on_policy_wdl_viz/outputs/stage_intervention_plateau_p60_math500.png`
- `analysis/on_policy_wdl_viz/outputs/stage_intervention_plateau_p60_math500.pdf`
- `docs/joint_training/courses/on-policy-wdl-overleaf/images/stage_intervention_plateau_p60_math500.png`
- `docs/joint_training/courses/on-policy-wdl-overleaf/images/stage_intervention_plateau_p60_math500.pdf`

The main plot compares the Stage 1 continuation curve with a Stage 2
intervention curve. Stage 2 is shifted so its step 0 equals the selected Stage 1
source checkpoint step. For example, the beta 0.0 chain uses Stage 1 step 85 as
the intervention point, so Stage 2 validation step 35 appears at effective step
120.

Use this Overleaf snippet after regenerating the figure:

```tex
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{images/stage_intervention_math500.pdf}
\caption{Stage-2 intervention effect on MATH-500 mean@3. Stage 2 is aligned to
the selected Stage-1 checkpoint; the dashed vertical line marks the handoff from
single-model on-policy SFT to Model2-rollout fused-loss WDL-SFT.}
\label{fig:stage-intervention-math500}
\end{figure}
```
