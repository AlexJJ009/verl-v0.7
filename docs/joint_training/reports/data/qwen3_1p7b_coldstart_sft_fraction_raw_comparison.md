# Qwen3-1.7B Code Format Cold-start SFT Fraction Comparison

| run | benchmark | pass@1 | base pass@1 | plus pass@1 | extract fail |
|---|---:|---:|---:|---:|---:|
| Raw base | HumanEval+ | 0.00% | 0.00% | 0.00% | 164/164 (100.00%) |
| Raw base | MBPP+ | 0.00% | 0.00% | 0.00% | 378/378 (100.00%) |
| Raw base | LiveCodeBench | 0.00% | pending | pending | 880/880 (100.00%) |
| SFT 25% | HumanEval+ | 44.51% | 46.34% | 44.51% | 1/164 (0.61%) |
| SFT 25% | MBPP+ | 49.74% | 55.56% | 49.74% | 2/378 (0.53%) |
| SFT 25% | LiveCodeBench | 7.27% | pending | pending | 36/880 (4.09%) |
| SFT 50% | HumanEval+ | 46.95% | 52.44% | 46.95% | 0/164 (0.00%) |
| SFT 50% | MBPP+ | 53.44% | 62.96% | 53.44% | 2/378 (0.53%) |
| SFT 50% | LiveCodeBench | 6.48% | pending | pending | 34/880 (3.86%) |
| SFT 100% | HumanEval+ | 48.78% | 53.66% | 48.78% | 2/164 (1.22%) |
| SFT 100% | MBPP+ | 51.59% | 59.52% | 51.59% | 2/378 (0.53%) |
| SFT 100% | LiveCodeBench | 6.25% | pending | pending | 24/880 (2.73%) |
