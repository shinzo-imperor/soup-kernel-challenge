Submission — Soup Kernel Challenge
Competition
Decentrathon 6.0
Project
Soup Kernel Challenge
GitHub Repository
https://github.com/shinzo-imperor/soup-kernel-challenge
Solution
The solution is located in the solution/ directory.
Implementation
• Entropy
• NF4
• RMSNorm
• SwiGLU
Notebook
Kaggle Notebook: [https://www.kaggle.com/code/fdgjkfgsdl/notebook692de55f5d]
Commit SHA
[d301c68ecfcc38371dde69a4b1364912e3d08e0a]
Results
[## Results on Kaggle T4 (from runs/result.json and runs/checks.json)

| | Base Soup | Ours | Ratio |
|—|—|—|—|
| Time of soup train, s | 79.9 | 78.2 | speedup 1.021x |
| Peak GPU memory, MiB | 14468 | 12506 | base/ours 1.157x |
| Eval loss (public rows) | 2.04967 | 2.04962 | -0.002 % vs base |

Own Triton kernels take 6.3 % of the training time (check 4, steps 3+).

- [x] All 10 checks of harness/check_rules.py passed
- [x] 79 Soup regression tests passed with our patches (check 10)]
