# Reviewer Comments — MLST-104723

**Manuscript:** "Flexible Genetic Algorithm for Quantum Support Vector Machines"
**Authors:** Nguyen Minh Duc, Vu Tuan Hai, Le Bin Ho, Lan Nguyen Tran
**Journal:** Machine Learning: Science and Technology (IOP Publishing)
**Decision:** Major Revision (DEC:ModRev:S)
**Revision deadline:** 01-Jun-2026
**Editor:** Jaya Srivastava (on behalf of Editor-in-Chief Kyle Cranmer)

---

## Table of Contents

- [Decision Letter Summary](#decision-letter-summary)
- [Referee 1 — Full Report](#referee-1--full-report)
  - [Summary and Overall Assessment](#summary-and-overall-assessment-r1)
  - [Major Comments](#major-comments-r1)
  - [Minor Comments](#minor-comments-r1)
- [Referee 2 — Full Report](#referee-2--full-report)
  - [Summary and Overall Assessment](#summary-and-overall-assessment-r2)
  - [Suggestions](#suggestions-r2)

---

## Decision Letter Summary

> The reviewer report(s) for your Paper are now complete. They recommend that you make some revisions before further consideration by Machine Learning: Science and Technology.

> Your final version and source files should be uploaded to the Author Centre by **01-Jun-2026**. If we do not receive your manuscript by this date, it may be treated as a new submission. Please let us know if you need more time.

**Required files for revision** (per IOP checklist):
- Author Response (point-by-point response to every reviewer's comment)
- Highlighted PDF (revised document with changes highlighted)
- Source File (clean Word/TeX/LaTeX format)
- Clean PDF version (unmarked copy)
- Supplementary material (optional)

---

# Referee 1 — Full Report

*Source: MLST-104723_review.pdf, dated 25 April 2026*

## Summary and Overall Assessment (R1)

> The authors propose GA-QSVM, a hybrid framework that employs Genetic Algorithms (GA) to automatically optimize quantum feature maps for Quantum Support Vector Machines (QSVMs). The framework introduces configurable metadata parameters governing both circuit expressibility and GA behavior, includes a normalization mechanism to stabilise the evolutionary search, and is evaluated on four benchmark datasets (Digits, Fashion, Wine, Breast Cancer). Transfer learning experiments, in which a circuit optimised on one dataset is applied to another, are also presented. The proposed approach is compared against classical SVM with an RBF kernel and conventional QSVMs using fixed feature maps (ZZFeatureMap, HEE).

> The topic is timely and the motivation is clear: fixed-ansatz QSVMs are unlikely to be universally optimal, and automating circuit design via evolutionary search is a well-motivated research direction. The paper is clearly written and the experimental setup is described with sufficient detail to be reproducible. However, the manuscript has several significant weaknesses that must be addressed before it can be recommended for publication, as detailed below.

---

## Major Comments (R1)

### R1-Major #1 — Lack of quantum hardware experiments and noise analysis

> The most substantial limitation of this work is that all experiments are conducted entirely on a classical simulator (an Intel Core i9-10920X CPU). No results on actual quantum hardware are reported. This is a critical gap for a paper claiming relevance to the NISQ era and to "future quantum machine learning applications." The very challenges that motivate the proposed approach — the need for shallow, hardware-efficient circuits and the sensitivity of performance to circuit topology — can only be meaningfully assessed on real quantum devices, where noise, gate error rates, decoherence, and qubit connectivity are physical constraints that a simulator cannot replicate. The paper does not even discuss the expected degradation of the GA-optimised circuits under realistic noise conditions, nor does it provide a noise simulation using established noise models (e.g., via Qiskit's AerSimulator with a device backend). The concurrent work on GA-QSVM that is published in this same journal (DOI: 10.1088/2632-2153/adb4bc) explicitly addresses hardware execution and parallelisation on superconducting QPUs, making the absence of any hardware-related analysis in the present submission more conspicuous.

**The authors should, at minimum:**

> a. Execute the best GA-optimised circuits (or a representative subset) on a real quantum device, such as an IBM Quantum or IonQ system accessible via cloud, and report the resulting classification accuracy;
>
> b. If hardware access is not possible, perform noise-aware simulations using a realistic device noise model and discuss the resilience of GA-evolved circuits relative to fixed-ansatz circuits; and
>
> c. Discuss explicitly how backend-specific constraints (qubit connectivity, native gate sets, transpilation overhead) would affect the circuits the GA produces, and whether the GA could be extended to incorporate these constraints into the fitness function.

> Without at least a noise simulation or a concrete discussion of hardware feasibility, the claim that GA-QSVM "highlights the potential of evolutionary strategies … for future quantum machine learning applications" (Abstract) is insufficiently substantiated.

---

### R1-Major #2 — Missing computational cost analysis

> The paper acknowledges in Section 6 that "the optimization process is still computationally expensive due to repeated kernel evaluations." However, no wall-clock runtimes, circuit evaluation counts, or comparisons of computational cost between GA-QSVM and baseline methods are reported anywhere in the paper. For a method whose practical viability depends on whether the search cost is tractable, this omission is significant. The authors should provide training time comparisons (in CPU-hours or equivalent) for GA-QSVM versus standard QSVM and classical SVM across the tested configurations, and discuss how runtime scales with the number of qubits, circuit depth, and population size.

---

### R1-Major #3 — Missing statistical analysis across multiple GA runs

> The results reported in Figure 6 are based on a single train/test/val split (subfigure 1) and k-fold cross-validation (subfigure 2). However, the GA itself is stochastic, and the authors note in Section 6 that "the stochastic nature of GA can lead to instabilities across runs." Despite this, no confidence intervals, standard deviations across independent GA runs, or statistical significance tests are reported. The hyperparameter sweep in Figure 4 also shows considerable variance across runs but this is not quantified. The authors should report mean and standard deviation over multiple independent GA runs (e.g., at least 5) for each dataset and configuration.

---

### R1-Major #4 — Test-set leakage in GA fitness evaluation

> As described in Section 4.1 and Algorithm 2, the GA optimizes circuits by repeatedly evaluating QSVM classification accuracy on a fixed train–test split, where the same test set appears to be reused throughout the evolutionary search. This risks implicit test-set leakage, as the GA may overfit circuit structure to the specific test split used during fitness evaluation, leading to optimistically biased accuracy estimates. Standard practice in machine learning would require either (i) a nested evaluation protocol, in which GA fitness is computed on a validation set and final performance is reported once on a held-out test set, or (ii) averaging fitness across multiple independent splits. The authors should clarify the data-splitting strategy used inside the GA loop and revise the evaluation protocol to avoid test leakage.

---

### R1-Major #5 — Missing comparison with alternative circuit search strategies

> While GA-QSVM is compared against fixed-ansatz QSVMs and classical SVMs, it is not compared against alternative circuit-search strategies with similar computational budgets, such as random circuit sampling, greedy or incremental circuit construction, or simpler local search methods. Without such baselines, it remains unclear whether the observed improvements arise from the genetic algorithm itself or from the increased flexibility of allowing non-fixed circuit structures. Including at least one stochastic or heuristic non-GA baseline would help substantiate the claim that evolutionary search provides a distinct advantage.

---

### R1-Major #6 — Superficial transfer learning analysis

> The transfer learning experiment (Section 5.3, Figure 7) is one of the more interesting aspects of the paper, but its analysis is superficial. Only a single source dataset (Digits) is tested, and the choice is motivated informally by it being "a complex dataset." There is no systematic study of which source datasets produce the most transferable circuits, no ablation comparing transferred circuits against randomly initialised circuits as a baseline, and no analysis of structural properties of the transferred circuits (e.g., entanglement depth, gate count) that might explain their transferability. The authors should either expand this analysis substantially or moderate the claims made about transfer learning in the Abstract and Conclusion.

---

### R1-Major #7 — Missing citations in Related Work

> The authors should revise Section 2 (Related Work) to include the four papers identified above. In particular:

> a. **Altares-López, Ribeiro & García-Ripoll (2021)** — "Automatic design of quantum feature maps" (Quantum Science and Technology, 6(4), 045015, DOI: 10.1088/2058-9565/ac1ab1) must be cited as the seminal paper in the GA-QSVM line of work. Its omission misrepresents the field's history, given that the authors already cite two of its direct successors (refs. [20] and [22]).

> b. **Chen & Chern (2022)** — "Generating quantum feature maps for SVM classifier" (arXiv:2207.11449) should be cited as an intermediate step in the same lineage, completing the chain from Altares-López et al. to Pellow-Jarman et al.

> c. **Moretti et al (2025)** — "Enhanced feature encoding and classification on distributed quantum hardware" (Mach. Learn.: Sci. Technol. 6 015056, DOI: 10.1088/2632-2153/adb4bc) should be cited in Section 2 and discussed in Section 6 in the context of the scalability and computational cost limitations.

---

## Minor Comments (R1)

### R1-Minor #1 — Sample size limitation

> The paper restricts evaluation to a maximum of 100 training and 100 testing samples for all datasets (Section 5.1). This is an extremely small subset of the full datasets (e.g., Fashion-MNIST has 70,000 samples). I would suggest the authors to discuss whether results generalise beyond this regime and whether the small sample size may artificially favour quantum kernels over classical ones.

---

### R1-Minor #2 — Unclear rationale for excluding Fashion-MNIST from GA-QSVM

> The paper states that Fashion-MNIST was excluded from GA-QSVM training "because of the huge number of Instances." This rationale is unclear: the paper uses only 100 training samples from each dataset. The real reason — likely the dimensionality after PCA (200 components) and the resulting qubit count — should be stated explicitly.

---

### R1-Minor #3 — Theoretical complexity claim clarification

> The complexity comparison in Section 1 states that the quantum dual problem requires O(M⁴·⁶⁷/ε²) circuit evaluations. I would suggest authors clarify whether this theoretical advantage is expected to be realised in the simulated regime of 100 samples and 7 qubits used in their experiments.

---

### R1-Minor #4 — Missing hyperparameter configuration table

> The metadata tuple M (Equation 12) is defined but the paper does not explain how its values were chosen for the main experiments (Figure 6). A table summarising the default hyperparameter configuration used in each experiment would improve reproducibility.

---

### R1-Minor #5 — Negative results in Figures 4(c) and 4(d) not discussed

> Figures 4(c) and 4(d) show that the number of CX gates and mutation probability have minimal impact on final fitness. This is an interesting negative result but is not discussed. The authors should comment on what this implies for the design of the GA, e.g., whether a simpler single-objective GA without tuning these parameters would suffice.

---

### R1-Minor #6 — Overstated generalization claim

> The paper claims that GA-QSVM "generalizes effectively across datasets" in the Abstract. This is an overstatement given that the transfer from Digits to Fashion only reaches 68.1% FQK accuracy, which is just marginally above the classical SVM baseline (65.7%) and well below the 88.4% achieved on the source task. I suggest the authors moderate the language to reflect the actual magnitude of the improvement.

---

### R1-Minor #7 — Grammatical errors and typographical issues

> Several grammatical errors and typographical issues are present throughout the manuscript. A thorough proofreading is recommended. In particular, double periods appear in Section 5.1 ("a classical computer..") and inconsistent spacing is present in several equations. I would suggest the authors carefully review the manuscript and fix all typos and inconsistencies before resubmission.

---

# Referee 2 — Full Report

*Source: Decision letter dated 19 May 2026*

## Summary and Overall Assessment (R2)

> The manuscript proposes a hybrid framework, GA-QSVM, which integrates Genetic Algorithms (GA) with Quantum Support Vector Machines (QSVM) to automatically optimize quantum feature-map circuits. The paper addresses an important challenge in quantum kernel methods: the dependence of QSVM performance on manually designed feature maps. The authors introduce a configurable evolutionary framework with adaptive normalization and evaluate the method on several benchmark datasets. The metadata-driven configuration of GA-QSVM is a useful contribution. The framework allows dynamic control over circuit depth, mutation probability, gate allocation, and population size.

> Overall, the topic is timely and relevant to current research in quantum machine learning and quantum kernel optimization. The manuscript is generally well organized and provides extensive experimental comparisons. Several suggestions would improve the paper.

---

## Suggestions (R2)

### R2 #1 — Strengthen transfer-learning section with complexity classification

> The transfer-learning section can be strengthened by introducing a simple dataset-complexity classification before transferring circuits. The authors could group datasets according to feature complexity or PCA variance decay, then study whether circuits evolved from high-complexity datasets transfer better to similar datasets. Finally, a brief analysis of circuit characteristics, such as depth and number of CX gates, could provide insight into why certain circuits transfer more effectively across datasets.

---

### R2 #2 — Additional Qiskit feature-map baselines

> The experimental comparison would be strengthened by including additional predefined feature-map circuits available in Qiskit, such as PauliFeatureMap, EfficientSU2, or TwoLocal ansatzes, to better evaluate whether the proposed GA-generated circuits provide advantages beyond standard fixed embeddings.

---

### R2 #3 — Computational cost as a limitation

> Add discussion of computational cost as a limitation.

---

### R2 #4 — Reproducibility improvements

> The reproducibility of the experiments can be improved by clarifying the random seed settings, train/test split procedure, and GA initialization process. The authors should also specify whether PCA was fitted only on the training data before being applied to the test data. Finally, a brief explanation of how hyperparameters such as circuit depth, mutation probability would strengthen the experimental methodology.

---

# Cross-Reference: Topic Overlap Between Reviewers

Một số chủ đề được cả hai reviewer đề cập với góc nhìn bổ sung cho nhau:

| Topic | Referee 1 | Referee 2 |
|---|---|---|
| **Transfer learning analysis** | Major #6 — yêu cầu phân tích đa dataset source, baseline random, structural analysis | #1 — yêu cầu complexity classification (PCA variance decay), depth/CX analysis |
| **Computational cost** | Major #2 — yêu cầu wall-clock numbers, scaling analysis | #3 — yêu cầu discussion as limitation |
| **Statistical rigor & reproducibility** | Major #3 — yêu cầu mean±std qua ≥5 runs | #4 — yêu cầu seed settings, GA init, **PCA fitting protocol**, hyperparameter justification |
| **Baseline comparisons** | Major #5 — yêu cầu random/greedy circuit search baselines | #2 — yêu cầu PauliFeatureMap, EfficientSU2, TwoLocal baselines |

**Unique to Referee 1 only:** R1-Major #1 (hardware/noise), R1-Major #4 (test-set leakage), R1-Major #7 (citations), R1-Minor #1–7.

**Unique to Referee 2 only:** PCA fitting protocol concern (within R2 #4) — methodological red flag không có trong Referee 1.

---

# Action Items Summary (for quick scan)

## 🔴 Critical — must address fully

- R1-Major #1: Add noise simulation (preferably + real hardware execution)
- R1-Major #3: Multi-run statistics (≥5 independent GA runs, report mean±std)
- R1-Major #4: Fix test-set leakage via nested CV
- R1-Major #7: Add 3 missing citations (Altares-López 2021, Chen & Chern 2022, Moretti 2025)
- R2 #4 (PCA part): Audit and confirm PCA is fitted only on training data

## 🟡 High priority — must address adequately

- R1-Major #2 + R2 #3: Report computational cost (wall-clock, evaluation counts, scaling)
- R1-Major #5: Add non-GA baselines (random, greedy circuit search)
- R1-Major #6 + R2 #1: Either expand transfer learning analysis OR moderate claims
- R2 #2: Add Qiskit baselines (PauliFeatureMap, EfficientSU2, TwoLocal)
- R2 #4 (other parts): Document seeds, splits, GA init, hyperparameter justification

## 🟢 Medium / Low — quick textual fixes

- R1-Major #1c: Discuss backend constraints qualitatively
- R1-Minor #1: Discuss small sample size implications
- R1-Minor #2: Clarify real reason for Fashion-MNIST exclusion (qubit count, not sample count)
- R1-Minor #3: Clarify whether O(M⁴·⁶⁷/ε²) advantage realized in their regime
- R1-Minor #4: Add hyperparameter configuration table
- R1-Minor #5: Discuss negative results in Figures 4(c)–(d)
- R1-Minor #6: Moderate "generalizes effectively" claim in Abstract
- R1-Minor #7: Proofread and fix typos

---

*Tài liệu này trích xuất toàn bộ comments từ Referee 1 (file MLST-104723_review.pdf ngày 25/04/2026) và Referee 2 (trong decision letter ngày 19/05/2026 từ Jaya Srivastava). Mọi quote đều giữ nguyên văn từ nguồn gốc.*
