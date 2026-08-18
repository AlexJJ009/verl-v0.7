# Layered Trajectories

This figure treats training history as a set of trajectories rather than a table of isolated scores. Three aligned panels preserve each benchmark's natural scale, while a shared horizontal rhythm makes changes in slope, plateaus, and late-stage regression directly comparable.

Color encodes initialization and cold-start data fraction; line structure encodes the reverse-SFT setting. This separates the two experimental variables without multiplying visual decoration. Raw initialization remains neutral gray, while increasingly complete cold-start supervision moves through blue, violet, and red.

The composition prioritizes legibility under high curve density. Restrained grids, repeated markers, generous margins, and a single shared legend keep all eight curves visible in one figure without hiding crossings or local peaks. Every stroke, interval, and label is calibrated for careful comparison rather than presentation drama.

The final artifact should feel meticulously crafted and technically exact: a research figure made through painstaking attention to hierarchy, spacing, and reproducibility. The visual language is quiet enough for a paper, but distinct enough that the experimental structure can be understood before reading the caption.

## Data Scope

- Model/task: Qwen3-1.7B, KodCode CTX8K, code Stage1.
- Metrics: online validation pass@1 for HumanEval+, MBPP+, and LiveCodeBench.
- Curves: raw initialization and cold-start SFT at 25%, 50%, and 100%, each with `beta=0.0` and `beta=0.1`.
- Steps: 0 through 150 at the recorded validation intervals.
