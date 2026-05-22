# Internal Reviewer Action Plan

This document maps every reviewer comment to our internal stance and concrete actions. It intentionally does not include draft response-letter text.

## Global Revision Strategy

- Clarify the two-stage protocol: GA search first, frozen-circuit benchmark second.
- Rename the GA fitness split from `test` to `search-validation`.
- Make GA-PQK the primary clean quantitative result.
- Present GA-FQK as a stability/candidate-sensitivity diagnostic if post-hoc candidate inspection was used.
- Add lightweight tables/analyses where possible.
- Moderate claims that cannot be supported without expensive reruns.

## Referee 1

### R1-Major #1: Hardware Experiments and Noise Analysis

**Stance:** Partially agree. Real hardware is likely not feasible within revision time, but the paper needs a hardware-feasibility discussion. A small representative noise simulation would help if time allows.

**Internal actions:**

- Must add a "Hardware feasibility and noise" paragraph in Discussion/Limitations.
- Must weaken Abstract/Conclusion claims about NISQ or future quantum applications.
- Discuss qubit connectivity, native gates, transpilation overhead, and noise sensitivity.
- Optional: run Qiskit Aer noise simulation for 1-2 representative circuits only.
- Do not promise real-hardware execution unless access/runtime is realistic.

### R1-Major #2: Computational Cost Analysis

**Stance:** Agree. This can be addressed mostly from logs/configs without rerunning full experiments.

**Internal actions:**

- Add formula: `cost ≈ n_circuit × n_generation × cost(QSVM fitness)`.
- Add default example: `n_circuit=20`, `n_generation=200` gives up to 4000 fitness evaluations for one GA search.
- Add runtime/evaluation-count table if logs are available.
- Discuss scaling with qubits, depth, population size, generation count, benchmark seeds, and dataset size.
- Add computational cost as a limitation in Conclusion.

### R1-Major #3: Missing Statistical Analysis Across Multiple GA Runs

**Stance:** Partially agree. We can add mean/std over repeated benchmark splits, but should not claim independent GA-run statistics if they were not run.

**Internal actions:**

- Add mean ± std over benchmark seeds 100-109 where raw notebook outputs already exist.
- Label this explicitly as "std over benchmark splits", not "std over independent GA runs".
- For FQK, add candidate-sensitivity table if multiple candidate circuits/configs were inspected.
- If time allows, run 2-3 independent GA reruns only for a small representative case.
- Otherwise, state independent GA-run variance as a limitation/future work.

### R1-Major #4: Test-Set Leakage in GA Fitness Evaluation

**Stance:** Strongly address by clarification. Current wording invites the criticism because the GA fitness split is called `test`. For GA-PQK main results, final benchmark test splits are separate from GA evolution. For FQK, be transparent if best candidate was selected after benchmark inspection.

**Internal actions:**

- Replace `D_test` with `D_val` or `D_search-val` in Method, Algorithm, and Eq. 8.
- Add final benchmark protocol paragraph before Figure 6.
- State that scaler/PCA are fitted only on train for each split.
- PQK: present as the main clean result.
- FQK: if selected by benchmark best, present as diagnostic/best-case or report all inspected candidates.

### R1-Major #5: Alternative Circuit Search Strategies

**Stance:** Agree in principle. Full greedy/local search may be expensive. Minimal defensible addition is random circuit search with the same gate pool and constraints.

**Internal actions:**

- Add random circuit search baseline if time allows.
- Same gate pool, depth/CX constraints, normalization, and fitness function as GA.
- Ideally use matched budget `B = n_circuit × n_generation`; if too slow, use a reduced budget and state it clearly.
- Greedy/local search can be discussed as future work if not run.
- Avoid adding too many new baselines unless runtime is manageable.

### R1-Major #6: Superficial Transfer Learning Analysis

**Stance:** Agree enough to revise. Either expand modestly or moderate claims. Best low-cost fix: PCA complexity classification + circuit structure table + reduced transfer-learning claims.

**Internal actions:**

- Soften "generalizes effectively across datasets".
- Use wording like "preliminary transferability" or "suggests transfer in selected settings".
- Add table: dataset, PCA components for 95% variance, selected circuit depth, CX count, accuracy.
- If no new experiments are run, explicitly call transfer learning preliminary.

### R1-Major #7: Missing Citations

**Stance:** Agree. Easy fix.

**Internal actions:**

- Add Altares-Lopez, Ribeiro & Garcia-Ripoll 2021.
- Add Chen & Chern 2022.
- Add Moretti et al. 2025.
- Mention Moretti again in hardware/scalability limitations.

### R1-Minor #1: Sample Size Limitation

**Stance:** Agree.

**Internal actions:**

- Add small-sample limitation in Section 5.1 or Discussion.
- State that 100/100 is chosen for simulator tractability.
- State that conclusions are small-data quantum-kernel benchmarks, not large-scale performance claims.
- Note that small samples may favor kernel methods and may not extrapolate to full datasets.

### R1-Minor #2: Fashion-MNIST Exclusion Rationale

**Stance:** Agree. Current explanation is weak.

**Internal actions:**

- Replace "huge number of Instances" with "large PCA dimensionality and qubit requirement".
- Mention that Fashion-MNIST needs around 200 PCA components for 95% variance, far beyond the simulated qubit budget.

### R1-Minor #3: Theoretical Complexity Claim Clarification

**Stance:** Agree. Do not imply theoretical advantage in the simulated regime.

**Internal actions:**

- Add one sentence after the Introduction complexity discussion.
- Clarify that the complexity discussion motivates the field, but the experiments do not claim speedup or asymptotic advantage.

### R1-Minor #4: Hyperparameter Configuration Table

**Stance:** Agree. Easy and important.

**Internal actions:**

- Add default metadata/config table near Section 5.1 or 5.2.
- Include qubits, population size, generations, depth, CX count, mutation probability, train/test sizes, benchmark seeds, kernel type.
- Include the exact main command/config if appropriate.

### R1-Minor #5: Negative Results in Figures 4(c) and 4(d)

**Stance:** Agree.

**Internal actions:**

- Add 2-3 sentences after Figure 4.
- Say that within the tested range, depth and population size dominate fitness improvement more than CX count or mutation probability.
- Do not overclaim that CX count or mutation probability never matter.

### R1-Minor #6: Overstated Generalization Claim

**Stance:** Agree. Moderate.

**Internal actions:**

- Abstract: replace "generalize effectively" with "show preliminary transferability".
- Conclusion: similarly soften.
- Make clear that transfer learning is exploratory, not a broad cross-dataset generalization proof.

### R1-Minor #7: Grammar and Typos

**Stance:** Agree.

**Internal actions:**

- Full proofreading pass.
- Fix duplicated punctuation such as "classical computer..".
- Fix Figure 6 typo "RPF" to "RBF".
- Check equation spacing and notation consistency.

## Referee 2

### R2 #1: Transfer Learning with Complexity Classification

**Stance:** Agree. This overlaps with R1-Major #6 and is a good lightweight addition.

**Internal actions:**

- Add PCA-based dataset complexity classification.
- Suggested buckets: low/medium/high based on components required for 95% variance.
- Add circuit descriptors: depth, CX count, total gate count if easy.
- Keep transfer claims preliminary.

### R2 #2: Additional Qiskit Feature-Map Baselines

**Stance:** Partially agree. Add at least one fixed feature-map baseline if feasible. Be careful with EfficientSU2/TwoLocal because they are ansatz circuits, not direct feature maps unless data encoding is defined.

**Internal actions:**

- Feasible addition: PauliFeatureMap.
- Consider EfficientSU2/TwoLocal only if the data-encoding interpretation is defensible and easy.
- If not added, explain that Z/ZZ are the primary standard feature-map baselines and leave broader ansatz baselines for future work.
- Adding PauliFeatureMap would strengthen the revision.

### R2 #3: Computational Cost as a Limitation

**Stance:** Agree. Same as R1-Major #2.

**Internal actions:**

- Add evaluation-count/runtime discussion.
- Explicitly identify repeated kernel evaluation during GA search as the main practical limitation.
- Add cost limitation in Conclusion.

### R2 #4: Reproducibility Improvements

**Stance:** Agree. Mostly textual and important.

**Internal actions:**

- Add seed table: GA search seed if fixed, data split seed(s), benchmark seeds 100-109.
- Add split sizes per dataset.
- Describe GA initialization: random circuits sampled from gate pool under metadata constraints.
- Add hyperparameter table.
- Explicitly state scaler/PCA are fitted only on training subset for each split.

## Highest-Priority Internal Checklist

### Must Fix in Paper

- Rename GA fitness split from `test` to `search-validation`.
- Add two-stage protocol: GA search then frozen-circuit benchmark.
- Add preprocessing statement: split first, scaler/PCA fit train only.
- Make GA-PQK the primary clean result.
- Present GA-FQK as stability diagnostic if best candidate was selected by benchmark inspection.
- Add missing citations.
- Add hyperparameter/default config table.
- Moderate transfer/generalization claims.
- Add computational cost discussion.

### Should Add if Time Allows

- Mean ± std over benchmark seeds for Figure 6.
- FQK candidate-sensitivity table.
- Random circuit search baseline.
- PauliFeatureMap baseline.
- Representative noise simulation.
- Runtime table from logs.

### Should Avoid Promising

- Full real-hardware execution.
- Full Figure 6 rerun over five independent GA searches.
- Strong claim that FQK best-case result is an unbiased final estimate.
- Strong claim that transfer learning generalizes broadly across datasets.

## Recommended Framing for the Revision

The revised manuscript should make the contribution narrower but more defensible:

- GA-QSVM is a configurable circuit-search framework for quantum kernels.
- In the simulator-limited small-data regime, GA-PQK is the main stable result.
- FQK is more candidate-sensitive and should be reported as diagnostic/stability analysis if selected post hoc.
- The method is computationally expensive and currently simulator-bound.
- Hardware-aware and noise-aware GA fitness should be framed as future work.
