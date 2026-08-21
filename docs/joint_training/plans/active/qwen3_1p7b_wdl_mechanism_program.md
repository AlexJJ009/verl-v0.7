# Qwen3-1.7B On-Policy WDL Mechanism Program

- Status: **ACTIVE RESEARCH DESIGN / DYNPERM 2x4 P60 MATRIX PREPARED** —
  theory-derived Math-first mechanism matrix. Fixed-Model1 arms are prepared
  separately. Dynamic Permutation core code was merged into the formal training
  branch at `8209576c04d89c7d778a249e8458c608f747c764`, with final focused CPU
  evidence (`114 passed`) and candidate-bound Job 146 8xL40S FSDP engineering
  smoke `PASS` (`formal_experiment=false`). Formal DynPerm training has not
  been launched. The approved execution design now crosses Standard C and its
  fixed-Model1 factorial edge with `rho={0,0.25,0.5,1}` directly to P60 through the shared
  `DYNPERM_ENABLED` / `DYNPERM_RHO` interface. The manifest remains
  `launch_allowed=false` and requires a separate candidate-bound P60 receipt.
- Created: 2026-08-16
- Primary method result: `qwen3_1p7b_math_stage123.md`
- Result attachment: `../../reports/qwen3_1p7b_math_stage123_matrix_results_20260723.md`
- Baseline/fixed-Model1 context: `qwen3_1p7b_on_policy_sft_baseline_extension.md`
- Canonical Feishu intervention protocol:
  [Dynamic Entropy-Matched Weak-Structure Ablation](https://ocnwds5io8yp.feishu.cn/docx/NEIvdnwU0o0vszxi2wycfcTHnjd)
- SFT precursor paper:
  [Weak-Driven Learning: How Weak Agents Make Strong Agents Stronger](https://arxiv.org/abs/2602.08222)
- Review record:
  [OpenReview forum](https://openreview.net/forum?id=WAqz1qihuI)

## 1. Decision

The same-entropy Dynamic Permutation experiment should be run. It is a direct
answer to the reviewer and a valid intervention on weak-logit token assignment.
It is not, by itself, a proof that Model1 is replaceable, and it does not fully
separate semantic identity from scalar gradient amplification after fusion.
Here, "same entropy" means **weak-entropy-preserving within the same forward**;
it does not mean that fused entropy is matched across C, D0, and DynPerm runs.
A separate scalar temperature/target-gradient control is required for that
reviewer-facing claim.

The mechanism program therefore has four successive questions:

1. **Is a trainable Model1 necessary?** Compare joint Model1, fixed Model1, and
   matched-scale no-weak D0.
2. **Is token assignment necessary?** Run target-preserving, step-resampled
   non-target DynPerm while keeping both models trainable.
3. **Is semantic identity necessary beyond fusion geometry?** Manipulate or
   match the weak/strong cross-distribution affinity while preserving the weak
   entropy and value multiset.
4. **Can a deterministic low-dimensional generator replace online Model1?**
   Build the surrogate only after Questions 1--3 identify which statistics are
   sufficient, then reverse the predicted mechanism.

The first mechanism claim should not be "Model1 is a hidden teacher." The more
precise working account is:

> On-policy WDL combines reward-projected hard self-training at the data layer
> with a trainable product-of-experts coupling at the logit layer. Model1
> changes the fused training objective's local geometry; when it is trainable,
> both models co-adapt to the same verifier-selected trajectories.

## 2. Evidence that the mechanism program must explain

The current Math causal-P60 result provides the starting facts, not a completed
mechanism proof:

- C uses `0.2 z1 + 0.8 z2`; D0 uses the matched-scale `0.8 z2` control.
- At the single training seed and online strict-scorer `n=3` endpoint,
  `C - D0 = +3.408 pp` Math-7 mean@3 and `+2.084 pp` exact pass@3; all seven
  Math datasets are directionally positive.
- In C, Model1 improves from `39.02%` to `71.04%` and Model2 from `42.61%` to
  `70.80%`. In D0, frozen Model1 stays at `38.80% -> 38.77%`, while Model2
  improves from `42.50%` to `67.39%`.
- C uses Model2-only rollout. Model1's large gain therefore cannot be explained
  by Model1 sampling its own successful trajectories.
- These are still single-seed online estimates. Common frozen evaluation,
  paired uncertainty, pass@k/diversity, and replication remain necessary for a
  publication-level efficacy claim.

Any useful theory must predict at least the C--D0 gap, the early acceleration of
C, and the fact that both trainable branches can improve.

## 3. Exact loss identities

### 3.1 Fused logits are a product of experts

Let

$$
z_m=(1-\lambda)z_1+\lambda z_2,
\qquad
p_i=\operatorname{softmax}(z_i).
$$

Then the fused distribution is exactly

$$
p_m(v)
=
\frac{p_1(v)^{1-\lambda}p_2(v)^\lambda}
{C_\lambda(p_1,p_2)},
\qquad
C_\lambda(p_1,p_2)
=
\sum_v p_1(v)^{1-\lambda}p_2(v)^\lambda.
$$

For $0\leq\lambda\leq1$, `C_lambda` is a Chernoff affinity. Define

$$
D_\lambda(p_1,p_2)=-\log C_\lambda(p_1,p_2)\ge 0.
$$

This statement is about the softmax distributions and is unchanged by adding
a constant to either branch's logits.

For any target distribution $q$---a one-hot SFT label or a soft OPD teacher---
the fused cross-entropy has the exact decomposition

$$
\operatorname{CE}(q,p_m)
=
(1-\lambda)\operatorname{CE}(q,p_1)
+\lambda\operatorname{CE}(q,p_2)
-D_\lambda(p_1,p_2).
$$

This identity gives a cleaner theoretical starting point than an unconstrained
residual expansion:

- both branches fit the same target distribution;
- the coupling is **negative Chernoff divergence**, not a positive KL term;
- subject to the target cross-entropies constraining the supervised
  coordinates, the residual coupling does not impose full-distribution
  imitation and may favor lower weak/strong affinity in the remaining mass,
  including the non-target tails.

At $\lambda=1/2$, $D_\lambda$ is the Bhattacharyya distance. This does not prove
better generalization or global convergence. It identifies the exact coupling
that mechanism experiments must manipulate.

### 3.2 Local gradient amplification is a consequence, not the full theory

For a positive token target $y$,

$$
\frac{\partial L}{\partial z_1}
=(1-\lambda)(p_m-e_y),
\qquad
\frac{\partial L}{\partial z_2}
=\lambda(p_m-e_y).
$$

For matched-scale D0, let

$$
p_{D0}=\operatorname{softmax}(\lambda z_2).
$$

The target-gradient amplification ratio for Model2 is therefore

$$
A_y^{C/D0}
=
\frac{1-p_m(y)}{1-p_{D0}(y)}.
$$

For a non-target coordinate $k\ne y$, the corresponding ratio is

$$
A_k^{C/D0}
=
\frac{p_m(k)}{p_{D0}(k)}.
$$

The precursor SFT theory proves sufficient local conditions under which these
quantities increase. It does not establish that the amplified direction is
useful, that the relevant tokens carry semantics, or that a multi-step adaptive
trajectory improves held-out accuracy. Those become empirical hypotheses here.
These are same-state, per-logit-coordinate ratios. Parameter-gradient and
optimizer-update amplification additionally depends on each model's Jacobian,
gradient clipping, optimizer state, and the trajectory distribution.

### 3.3 The defensible self-distillation interpretation is at the data layer

For rollout policy $\pi_{\mathrm{roll}}$ and binary verifier reward, define the
reward-projected successful-trajectory distribution

$$
q^+(\tau\mid x)
=
\frac{\pi_{\mathrm{roll}}(\tau\mid x)\mathbf{1}[R(\tau)=1]}
{\sum_{\tau'}\pi_{\mathrm{roll}}(\tau'\mid x)\mathbf{1}[R(\tau')=1]}.
$$

The population positive-SFT objective is cross-entropy from $q^+$ to the
current policy and is therefore equivalent, up to the entropy of $q^+$, to

$$
\min_\theta D_{\mathrm{KL}}(q^+\Vert\pi_\theta).
$$

This interpretation is per iteration and conditional on a non-empty successful
set. It treats the sampled-and-filtered $q^+$ as a frozen target: no gradient
passes through sampling, the verifier indicator, or its normalizer. All-wrong
groups and any fallback or reverse-SFT branch are outside this KL identity. In
finite groups, normalizing over the observed successful samples also weights
prompts by discovery probability; it is only an empirical approximation to the
population conditional distribution above.

This supports the restricted description **reward-projected hard
self-distillation** or **verifier-gated self-training**. In the current causal
P60 setting, Model2 proposes trajectories, the verifier selects successful
ones, and both Model1 and Model2 fit those hard trajectories through the fused
objective. This data flow can explain why Model1 improves.

It does not make Model1 a standard teacher:

- no KL makes Model2 imitate Model1's soft distribution;
- the logit coupling contains a negative, not positive, inter-model divergence;
- with reverse-SFT (`beta > 0`), the signed objective is no longer a KL to a
  valid probability target.

Safe paper wording is:

> The positive branch admits a reward-projected hard self-distillation
> interpretation, whereas the weak model acts through fused-logit geometry
> rather than as an explicit teacher distribution.

### 3.4 A unified explanation for SFT/SFD and OPD

The product-of-experts identity holds for any supervision target $q$:

- **Offline SFT:** $q$ is a fixed one-hot demonstration distribution. WDL is
  coupled hard-label training; there is no on-policy distillation claim.
- **On-policy SFT/SFD:** $q$ is the empirical reward-projected rollout
  distribution. The data layer is hard self-distillation; WDL adds the same
  product-of-experts coupling.
- **OPD:** $q$ is an explicit teacher distribution evaluated on student states.
  Up to the constant $H(q)$, WDL becomes two weighted forward-KL distillation
  terms from $q$ to the branches minus the same Chernoff divergence term.

This suggests that WDL is a **supervision-agnostic coupling operator**, while
SFT/SFD/OPD determines where $q$ comes from. If WDL helps both on-policy SFT and
OPD, the shared explanation should be the coupling/geometry, not the presence
of distillation alone.

## 4. Mechanism hypotheses

| Hypothesis | Claim | Distinguishing prediction |
| --- | --- | --- |
| H0: longer/harder SFT | Model1 merely keeps ordinary CE active | D0, LR/step/entropy-matched single-model controls catch up; weak assignment manipulations add nothing |
| H1: scalar shape | Weak entropy, target confidence, or tail spectrum is sufficient | target-preserving random DynPerm and a no-WM1 shape generator remain close to C |
| H2: cross-rank geometry | Benefit comes from alignment of weak and strong non-target ranks, through `C_lambda` and fused gradient magnitude | same-rank, random, and anti-rank interventions produce the preregistered affinity/gradient telemetry ordering; whether endpoint accuracy follows is an empirical test |
| H3: token-specific semantics | Particular weak logits identify useful semantic confusions/hard negatives beyond scalar/rank geometry | real assignment beats affinity-matched or rank-bin permutation at matched fused target probability/gradient strength |
| H4: adaptive co-training | Online Model1 updates and Model1/Model2 co-adaptation are necessary | joint C beats fixed-M1; fixed/cached/synthetic controls approach D0 |
| H5: rollout transfer | Main gain comes from Model2's better successful trajectories rather than fused training geometry | D0 trained on C rollouts catches up; C trained on D0 rollouts loses most of its advantage |

The mechanism may contain more than one component. The experiment program
should estimate their increments rather than force a single-label explanation.

## 5. What DynPerm does and does not identify

The canonical intervention uses a step-resampled, target-preserving non-target
permutation $P_{y,t}$:

$$
\widetilde z_1=P_{y,t}z_1,
\qquad
z_m^{perm}=(1-\lambda)\widetilde z_1+\lambda z_2.
$$

It exactly preserves, within the current forward:

- weak entropy;
- weak target-token probability;
- the complete weak logit/probability value multiset;
- both gradient paths, because
  $\partial L/\partial z_1=(1-\lambda)P_{y,t}^{\top}
  \partial L/\partial z_m^{perm}$.

It breaks the correspondence between weak non-target values and token identity.
It does **not** preserve:

- the mixed entropy;
- $C_\lambda(P_{y,t}p_1,p_2)$;
- fused target probability;
- Model2 gradient magnitude;
- future trajectory entropy after the intervention causes the runs to diverge.

Therefore the primary `rho=0` versus `rho=1` comparison estimates whether
continually preserving the real token assignment matters to the whole adaptive
training procedure. It cannot, alone, distinguish semantic dark knowledge from
rank-alignment-induced gradient strength.

This is the central extension to the existing same-entropy plan: after plain
DynPerm, add a control that also matches the relevant **cross-model fusion
geometry**.

The repository implementation note is
`../../specs/dynamic_permutation_mvp.md`. The current MVP uses a stateless keyed
cyclic selected-set with a non-zero rotation over exactly
$k=\lfloor\rho(V-1)\rfloor$ non-target coordinates, processes token rows in
bounded chunks with `O(row_chunk_size * k)` index memory, and derives the mapping
from explicit training identities rather than process-global RNG. Runtime audits
are bounded counters plus sampled entropy/multiset checks; exhaustive invariant
checks remain in CPU fixtures.

As of 2026-08-20, the Delivery evidence is code and CPU-only tests. The
candidate-bound Slurm GPU/FSDP smoke, independent review, and PR/CI evidence are
still required before merge readiness. This plan must stay active; code delivery
does not mark the mechanism experiment complete.

Two useful boundary facts follow. First, replacing weak logits with a uniform
vector is already equivalent, up to an additive logit constant, to matched-scale
D0: `softmax((1-lambda)c + lambda*z2) = softmax(lambda*z2)`. Second, DynPerm
does not implement the reviewer's stronger fused-entropy-matched control. That
requires a separate Model2-only temperature or target-margin transform.

## 6. Theory-guided interventions

### 6.1 M0: no-new-training diagnostics

Use existing P0/P25/P40/P55/P60 C and D0 checkpoints and stored validation
samples. Record per response token and by confidence/disagreement bin:

1. weak/strong/fused/D0 target probability and target rank;
2. weak/strong entropy and centered logit norm;
3. `log C_lambda` / Chernoff distance;
4. C-vs-D0 target and non-target amplification ratios;
5. weak/strong top-k overlap and non-target rank correlation;
6. Model1/Model2/fused correctness overlap, Model1-only and Model2-only wins;
7. format, EOS, truncation, response length, and answer diversity.

On the same frozen tokens, also compute two Model2-only counterfactuals before
launching new training:

8. `EntropyMatch`: solve a scalar temperature so the transformed D0
   distribution matches C's fused entropy;
9. `TargetGradMatch`: add a target-logit bias so its target probability, hence
   the one-hot target-coordinate gradient magnitude and aggregate non-target
   mass, matches C. This deliberately does not match the full non-target
   gradient vector.

These are diagnostic controls if their target statistics are read from C/WM1;
they become a WM1 replacement claim only after the controller is calibrated on
a disjoint split and no WM1 forward is used in the evaluated training run.

The preregistered diagnostic prediction is that early `A_y`, affinity, or
disagreement bins should predict where C later gains over D0. Correlation is a
mechanism diagnostic, not a causal endpoint claim.

### 6.2 M1: fixed-Model1 factorial edge

Use the already designed Math P60 arms:

- `C-joint`: real assignment, Model1/Model2 trainable;
- `C-fixed-M1-S1`: same Stage1 Model2 source, Model1 frozen;
- `D0`: matched-scale no-weak;
- optional CS0 fixed arm compared with matched A.

Interpretation:

- fixed $\approx$ joint $>$ D0: static weak guidance is sufficient;
- joint $>$ fixed $>$ D0: static guidance and co-adaptation both contribute;
- joint $>$ fixed $\approx$ D0: adaptive co-training is necessary;
- fixed or joint $\le$ D0: no positive weak contribution under that source.

Frozen Model1 still requires Model1 forward. It does not establish a no-WM1
method.

### 6.3 M2: canonical DynPerm

First validate the endpoints:

- `DynPerm-0`: real C;
- `DynPerm-100`: all non-target weak coordinates permuted independently per
  token and optimizer step, with deterministic seeds.

Run the frozen dose grid `rho={0,0.25,0.5,1}` directly to P60 for both Standard
C (Model1 trainable) and the matched fixed-Model1 factorial edge. This is a 2x4
factorial matrix with eight runs. `rho=0` keeps the feature enabled and is the
same-revision no-op control; `rho=1` is the full endpoint, while `rho=0.25/0.5`
identify whether any endpoint effect is monotone, threshold-like, or non-linear.
Use the P20/P30 checkpoints on each continuous P60 trajectory for early
diagnosis rather than launching separate truncated jobs.

The scheduling order is fixed-Model1 first, in dose order `0,1,0.25,0.5`, then
the same four Standard C runs at lower Slurm priority. All eight jobs may be
submitted together to the three-node `l40s` partition; each run owns one
exclusive 8xL40S node, so three run concurrently and the rest remain queued.
Scheduling priority is an operational policy, not an experimental variable.
Each job writes full logs and checkpoints on its executing node and relays only
candidate-bound admission, first-step, terminal, and bounded log-tail evidence
to the controller. The submission ledger records both the worker-local artifact
root and the controller receipt path; a failed relay fails that job rather than
silently claiming monitored completion.

Before any `sbatch`, the submitter validates all four rho receipts against the
same parent, recipe, image, horizon, and exact two-arm authorization, then
exercises worker-to-controller rsync from all three nodes. It submits all eight
jobs held; a submission error cancels only those new held job IDs, and the
matrix is released only after all eight submissions succeed. Terminal evidence
is transferred separately after the rest of the bounded evidence set and is
accepted only when the controller copy has the expected SHA-256.

The preregistered analysis order is:

1. fixed-Model1 endpoint contrast, `rho=0` versus `rho=1`;
2. fixed-Model1 four-dose curve;
3. the corresponding Standard C endpoint and curve;
4. the Model1-update-state by rho interaction.

A decreasing curve as rho increases supports dependence on weak-logit token
assignment. A larger degradation for trainable Standard C than fixed-Model1
supports a co-adaptive assignment mechanism. A flat curve rejects this
intervention under the frozen contract; it does not prove Model1 is replaceable.

For fast screening, the historical C run may serve as `rho=0` only after a
`rho=0` no-op equivalence test proves identical forward, backward, RNG, and
configuration behavior. Publication confirmation should use a common code
revision and matched seeds for both endpoints.

### 6.4 M3: directional same-entropy geometry controls

For each token, keep the target coordinate fixed and preserve the weak
non-target value multiset. Let the transformed weak tail be assigned to strong
tail ranks in three ways:

1. `AlignSort`: largest weak tail values paired with largest strong tail values;
2. `RandomPerm`: random target-preserving pairing;
3. `AntiAlignSort`: largest weak tail values paired with smallest strong tail
   values.

By the rearrangement inequality, these arms deliberately order the non-target
contribution to $C_\lambda$ from high to low, while preserving weak entropy and
the weak value multiset. With the target numerator fixed:

$$
C_\lambda\uparrow
\Longrightarrow
p_m(y)\downarrow
\Longrightarrow
1-p_m(y)\uparrow.
$$

This produces a directional **gradient-strength** test, not an endpoint
prediction from the Chernoff term alone. `AlignSort` should lower fused target
probability and enlarge the positive-token logit gradient; `AntiAlignSort`
should do the opposite. If endpoint gains also follow
`AlignSort > RandomPerm > AntiAlignSort` together with this preregistered
telemetry, scalar fusion geometry/gradient strength is a credible causal
component even though token semantics were destroyed. The negative Chernoff
term itself favors lower affinity, so endpoint ordering must be measured rather
than asserted from the decomposition.

Add `RankBinPerm` as the semantic control: partition tokens into narrow bins by
strong-logit rank and permute weak values only within each bin. This
approximately preserves `C_lambda`, fused target probability, and scalar
gradient strength while breaking exact token identity. If real C still beats
`RankBinPerm`, token-specific semantic assignment has evidence beyond rank
geometry.

`RankBinPerm` is admitted as a semantic causal control only if it simultaneously
achieves high permutation coverage/low fixed-point rate and matches the full
distributions of fused target probability and gradient ratio within
preregistered tolerances, including confidence-stratified bins. If narrow bins
retain too much identity or wide bins fail geometry matching, it remains a
diagnostic rather than a causal control.

The first implementation may use a short P20/P30 trajectory-separation pilot
for `AlignSort` and `AntiAlignSort`. Full P60 is admitted only if validity
metrics show the intended affinity ordering without format/runtime failure.

### 6.5 M4: no-online-WM1 scalar controller or surrogate

Do not train another full neural branch; that would recreate Model1 and fail the
replacement objective. Fit a small deterministic generator from a disjoint
calibration split after M1--M3 identify sufficient statistics. Candidate form:

$$
\widehat z_w
=G_{b(t),b(p_2(y)),b(H(p_2))}
\big(\operatorname{rank}(z_2), y\big),
$$

where `G` is a table or a few scalar functions that specify target confidence,
tail spectrum, and optional rank alignment by token position/confidence bins.
It has no online Model1 forward, parameters, optimizer, or semantic token
knowledge from a second network.

For hard SFT/SFD targets, the minimal candidate is not a learned network but a
target-margin controller. Let $q_0=\operatorname{softmax}(\lambda z_2)$ and add
a scalar bias $\delta$ only to target logit $y$. The controlled probability is

$$
q_\delta(y)
=
\frac{e^\delta q_0(y)}{e^\delta q_0(y)+1-q_0(y)}.
$$

To match a desired fused target probability $r$, the exact bias is

$$
\delta
=
\operatorname{logit}(r)-\operatorname{logit}(q_0(y)).
$$

With $q_0(y),r\in(0,1)$, this matches the target-coordinate CE gradient and
aggregate non-target gradient mass without importing WM1's token-specific tail:

$$
q_\delta(k)
=q_0(k)\frac{1-r}{1-q_0(y)},\qquad k\ne y.
$$

Therefore it preserves Model2's relative non-target ordering and changes every
non-target CE-gradient coordinate only by one shared scale factor. It does not
match C's complete logit-gradient vector, parameter-gradient direction, or
token-specific weak tail. Clamp $q_0(y)$ and $r$ away from zero/one in code.

The controller is a stop-gradient forward intervention: compute $q_0(y)$, $r$,
and $\delta$ from detached values and backpropagate only through the transformed
Model2 logits. Allowing a $q_0\rightarrow\delta$ gradient path can cancel or
redefine the intended training signal. A two-scalar version combines a detached
temperature and target bias to match both fused entropy and target probability
while retaining Model2's non-target ordering; infeasible or numerically unstable
matches must be logged and rejected.

First use the actual C statistics only as a frozen diagnostic; then fit $r$ or
$(r,H)$ as a function of detached Model2 confidence, step, and position on a
disjoint calibration split. The latter is a genuine no-online-WM1 training arm,
although it still uses prior WM1 traces if they supplied the calibration
targets. This one-hot construction does not directly establish the same
replacement for a soft-target OPD loss.

Required controls:

- calibration prompts, evaluation prompts, training prompts, and prompt
  templates are disjoint;
- parameter/storage/FLOPs disclosed;
- matched-scale D0;
- fixed/cached weak guidance if technically valid;
- a target-confidence-only and an entropy/spectrum version.

If calibration used Model1 traces, the allowed claim is "online Model1 is not
necessary," not "the method never uses weak-model information."
If rollout remains Model2-only, this surrogate replaces weak guidance in the
training objective only; it does not demonstrate replacement of a weak model in
a fused rollout policy.

### 6.6 M5: reverse experiments

Two reverse tests complete different causal paths.

**Geometry reversal.** Relative to `AlignSort`, `AntiAlignSort` reverses the
preregistered affinity, fused-target-probability, and logit-gradient-strength
telemetry. The geometry mechanism is supported only if those measured
quantities move as predicted and endpoint changes track the same direction; the
product-of-experts identity alone does not predict endpoint accuracy.

**Scalar-gradient reversal.** First determine from M0 whether C's fused target
probability is below or above matched-scale D0 in the relevant confidence bins.
If the calibrated target-margin controller closes part of the C--D0 gap by
moving $r$ in that observed C direction, run a magnitude-matched arm that moves
$r$ by the same feasible target-probability/target-gradient displacement in the
opposite direction; equal $|\delta|$ alone is not gradient-matched. For the
expected amplification case this is $\delta<0$ (lower target probability and
larger positive-token gradient) versus $\delta>0$ attenuation, but the sign must
be preregistered from M0 telemetry rather than assumed. Opposite telemetry and
an opposite learning-curve response provide a cleaner reverse test only if
target-gradient magnitude, pre-clip gradient norm, clip frequency, post-clip
norm, and optimizer update norm are audited. Otherwise it is an optimization
stress test rather than a causal reversal. It remains a hard-target SFT/SFD
test, not an automatic explanation of soft-target OPD.

**Rollout/training reversal.** Freeze one common rollout batch or manifest and
cross its source with its training objective:

| Rollout source | Training objective |
| --- | --- |
| successful manifest from the C checkpoint's Model2 proposer | C fused loss |
| the same C-proposer manifest | D0/model2-only loss |
| successful manifest from the D0 checkpoint's Model2 proposer | C fused loss |
| the same D0-proposer manifest | D0/model2-only loss |

If D0 trained on C's successful trajectories catches C, trajectory quality or
reward-projected self-distillation dominates. If C still wins on the same
trajectories, fused geometry contributes independently. Because online runs
diverge immediately, define each source by exact policy/checkpoint/step and
start with the same checkpoint, optimizer state, prompt shard, and one frozen
rollout manifest in one-step or short-horizon branches. Record positive rate,
all-wrong/all-correct group rate, length, format, and EOS. This estimates a local
data-source-by-objective interaction under fixed data, not the full online
mechanism; confirm with online endpoints only if the interaction is material.

A second directional test swaps the proposer: Model1 rollout with Model2
training versus Model2 rollout with Model1 training. This is lower priority
because proposal quality and rollout support change together.

### 6.7 M6: explicit-distillation bridge

Only after the preceding results, compare WDL with an explicit KL bridge on the
same successful prefixes:

- hard selected-SFT only;
- explicit Model2-to-Model1 KL;
- symmetric JS/deep-mutual-learning-style KL;
- WDL fused objective;
- WDL plus explicit OPD target, where available.

This test is not needed to call the positive branch reward-projected hard
self-distillation. It is needed if the paper wants to claim that the
logit-fusion mechanism itself approximates or improves standard distillation.

## 7. Minimal Math-first execution matrix

All first-wave runs use Qwen3-1.7B, the existing Stage1 source, the ordered
post-Stage1 3,840-prompt shard, Model2 rollout, `beta=0`, `lambda=0.8`, P60, and
the existing Math-7 online validation contract unless the arm definition
explicitly changes the rollout source.

| Wave | New training | Primary comparison | Gate to next wave |
| --- | --- | --- | --- |
| M0 | none | existing C/D0 checkpoints | telemetry can distinguish real, D0, and confidence bins |
| M1 | prepared fixed-M1 arms; resolve Slurm admission before treating them as running | joint/fixed/D0 | quantify static guidance vs co-adaptation |
| M2 P60 | 2x4 `rho={0,0.25,0.5,1}` matrix over Standard C and fixed-Model1 | dose crossed with Model1 update state | intervention validity passes and P20/P30/P60 trajectories are interpretable |
| M3 pilot | `AlignSort`, `AntiAlignSort`, optional `RankBinPerm` P20/P30 | directional affinity ordering | measured `C_lambda` and gradient ordering match theory |
| M3 confirm | only material pilot arms to P60 and second seed | geometry vs semantic identity | identify sufficient statistics |
| M4 | one calibrated target-margin/entropy controller; richer surrogate only if needed | controller vs C/fixed/D0 | controller approaches C without online WM1 |
| M5 | short same-rollout 2x2; online confirmation only if needed | data source x loss geometry | separate trajectory transfer from training geometry |

The immediate DynPerm batch is eight continuous P60 runs: four rho values for
fixed-Model1 Stage1 and the same four for Standard C. Fixed-Model1 receives
higher scheduling priority. The batch does not include M3 synthetic controls.

## 8. Metrics and validity contracts

### 8.1 Efficacy

- Model1-only, Model2-only, and fused Math-7 mean@n/pass@n;
- common frozen `n=8`, then selected checkpoints at `n=256`;
- paired prompt-level correctness and bootstrap intervals;
- second training seed for every conclusion-changing arm;
- format, EOS, truncation, length, diversity, and calibration;
- generated tokens, training tokens, FLOPs proxy, GPU-hours, and storage.

### 8.2 Intervention validity

For every transformed forward, log or assert:

- target coordinate unchanged;
- weak entropy difference within numeric tolerance;
- weak target-probability difference within tolerance;
- weak value-multiset checksum or sorted-value difference;
- permutation coverage/fixed-point fraction;
- deterministic seed derived from training seed, global step, sample, and token;
- Model1 and Model2 gradients present for joint arms;
- intended `C_lambda`/rank-correlation ordering for M3 arms.

For approximately matched controls such as `RankBinPerm`, report distributions
and confidence-stratified differences of fused target probability and gradient
ratio, not only global means. Every comparison also freezes code revision,
training seed, data order, prompt shard, and generated/selected-token accounting.

### 8.3 Mechanism telemetry

- `log C_lambda` and Chernoff distance;
- fused and D0 target probabilities;
- target/non-target gradient amplification ratios;
- per-branch pre-clip grad norm, update norm, clip frequency, and cosine;
- weak/strong top-k overlap and rank correlation;
- positive-trajectory rate and group composition;
- Model1/Model2 correctness overlap over training.

Endpoint accuracy without validity and mechanism telemetry is not sufficient to
interpret these experiments.

## 9. Result-to-claim map

| Observation | Supported interpretation | Claim that remains invalid |
| --- | --- | --- |
| fixed $\approx$ joint $>$ D0 | static weak guidance largely sufficient | no online Model1 needed |
| joint $>$ fixed $\approx$ D0 | adaptive co-training is central | weak entropy/shape alone is sufficient |
| real $>$ RandomPerm | real token assignment/alignment matters | token semantics, rather than affinity/gradient magnitude, caused the gain |
| real $\approx$ RandomPerm $>$ D0 | weak value spectrum/shape may be sufficient | Model1 is replaceable |
| Align $>$ Random $>$ Anti with predicted telemetry | cross-rank geometry and gradient strength are causal candidates | semantic information is irrelevant |
| real $>$ RankBin at matched affinity/gradient | token-specific semantic structure has evidence | every high weak logit is a meaningful hard negative |
| low-dimensional surrogate $\approx$ C | online WM1 can be replaced under the tested domain | universal no-WM1 equivalence |
| target-margin control closes the gap and sign reversal flips the curve | scalar target-gradient strength is a causal component for hard SFT/SFD | token-specific tails never matter; the same mechanism explains soft-target OPD |
| D0 on C rollouts catches C | trajectory transfer/self-training dominates | fused geometry has no effect in other regimes |
| C wins on shared rollouts | fused objective contributes beyond rollout quality | product-of-experts coupling alone explains long-run accuracy |

Null results require equivalence margins and confidence intervals; lack of a
significant difference is not affirmative proof of equivalence.

## 10. Literature anchors and boundaries

- [GKD / On-Policy Distillation, ICLR 2024](https://arxiv.org/abs/2306.13649):
  student-generated states plus explicit teacher distribution. It establishes
  that OPD predates 2026, but WDL has no equivalent teacher KL.
- [MiniLLM, ICLR 2024](https://arxiv.org/abs/2306.08543): reverse-KL on-policy
  distillation; relevant to sequence distribution mismatch, not an exact WDL
  objective.
- [Self-Distilled Reasoner / OPSD, 2026 preprint](https://arxiv.org/abs/2601.18734):
  one model with privileged teacher context and explicit token divergence.
- [SDPO, 2026 preprint](https://arxiv.org/abs/2601.20802): feedback-conditioned
  self-teacher turns rich feedback into dense supervision.
- [Rethinking OPSD for Thinking Models, 2026 preprint](https://arxiv.org/abs/2607.05184)
  and [Anti-Self-Distillation, 2026 preprint](https://arxiv.org/abs/2605.11609):
  teacher-like agreement can suppress high-entropy reasoning forks; these works
  motivate testing disagreement/anti-distillation rather than presuming that
  more imitation is always better.
- [U-OPSD, 2026 preprint](https://arxiv.org/abs/2608.06296): self-consistency
  constructs a pseudo-solution and a privileged self-teacher. It remains an
  explicit distillation method, unlike WDL logit fusion.
- [Born-Again Neural Networks / DKPP, ICML 2018](https://proceedings.mlr.press/v80/furlanello18a.html):
  the direct preserve-and-permute antecedent for separating value spectrum from
  class/token identity.
- [Deep Mutual Learning, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html):
  jointly updated peers with explicit prediction matching; useful precedent for
  co-adaptation, not an exact WDL baseline.

The 2026 self-distillation papers above are preprints unless a primary venue
record is later verified. They support hypotheses and terminology, not claims
of established consensus.

## 11. Immediate implementation boundary

The next coding task should implement only:

1. a reusable target-preserving permutation transform with `rho=0/1`;
2. deterministic per-step/sample/token seeds;
3. autograd-preserving gather/scatter semantics;
4. intervention validity assertions and `C_lambda`/gradient telemetry;
5. one Math P60 matrix that crosses Standard C versus fixed-Model1 at
   `rho={0,0.25,0.5,1}` through the shared `DYNPERM_ENABLED` / `DYNPERM_RHO`
   interface and retains an enabled no-op-equivalent `rho=0` path;
6. unit tests for entropy, target probability, multiset, gradient connectivity,
   determinism, and `rho=0` exact equivalence.

Do not implement the full M3--M6 matrix in the first patch. Their exact form is
conditioned on M0--M2 evidence.

2026-08-20 execution decision: do not launch separate P20/P30 jobs. The shared
causal-P60 entry accepts only the two public DynPerm knobs
`DYNPERM_ENABLED` and `DYNPERM_RHO`; the existing Standard C and fixed-Model1
wrappers continue to own Model1 update state. The three-node Slurm submitter
queues all eight P60 runs with fixed-Model1 first and lower-priority Standard C.
Every non-treatment model/data/optimizer/validation/seed/image setting remains
identical. Real execution remains fail-closed on the GON-34 engineering receipt
plus a separate candidate-, image-, and rho-bound P60 receipt. Worker-local
first-step and terminal receipts are relayed to the controller; relay failure is
terminal failure, not missing-but-assumed evidence.
