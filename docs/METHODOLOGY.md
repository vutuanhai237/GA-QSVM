# Experimental Methodology Clarification

This document summarizes the actual experimental protocol used for the GA-QSVM results and clarifies the distinction between GA search, final benchmark evaluation, and the different roles of PQK and FQK results.

## Main Message

The final benchmark test splits are not used during GA evolution. The confusion comes from the current code and manuscript terminology, where the split used inside GA fitness was sometimes called `test`. In the experimental protocol, that split should be understood as a search-validation split, not the final benchmark test set.

The experiment has two stages:

```text
Stage 1: GA circuit search
Full dataset
  -> fixed search split
     -> search train: train QSVM inside the GA fitness function
     -> search validation: compute GA fitness for circuit selection
  -> output: candidate circuits

Stage 2: final benchmark evaluation
Freeze selected circuit
  -> repeat over random benchmark seeds
     -> create a fresh train/test split
     -> fit scaler/PCA on benchmark train only
     -> train QSVM/SVM on benchmark train
     -> evaluate accuracy on benchmark test
  -> report mean and standard deviation
```

## Stage 1: GA Circuit Search

The GA search is run with a command of the form:

```text
/home/qsvm/temp/GA-QSVM/main.py --num-circuit 20 --num-generation 200 --qubits 6 --data digits --kernel pqk
```

During this stage, each candidate circuit is evaluated by training QSVM on a search-training subset and computing accuracy on a search-validation subset. This score is used as the GA fitness. The purpose of this stage is circuit discovery, not final performance reporting.

Important terminology correction for the paper:

- Current wording/code may call this subset `test`.
- In the paper, it should be called `validation`, `search-validation`, or `fitness-validation`.
- The final benchmark test set is separate from this concept.

Recommended paper wording:

```text
During GA evolution, each dataset is split into a search-training subset and a search-validation subset. The search-validation subset is used only to compute GA fitness and select candidate circuits during the search stage. It is not used as the final benchmark test set reported in the performance comparison.
```

## Stage 2: Final Benchmark Evaluation

After GA produces candidate circuits, the selected circuit is frozen. The frozen circuit is then evaluated across repeated random data splits. For each benchmark seed:

1. A train/test split is generated.
2. Preprocessing is fitted on the training subset only.
3. The trained preprocessing is applied to the test subset.
4. QSVM/SVM is trained on the benchmark training subset.
5. Accuracy is computed on the benchmark test subset.

This benchmark stage is used to estimate final performance.

Recommended paper wording:

```text
For final evaluation, the circuit discovered by GA is frozen before benchmark evaluation. We then evaluate the frozen circuit over repeated random train/test splits. In every split, scaling and PCA are fitted only on the training subset and then applied to the test subset.
```

## PCA and Preprocessing

The preprocessing order used in the benchmark notebooks is:

```text
raw data
  -> train/test split
  -> fit scaler on train
  -> transform train/test
  -> fit PCA on train
  -> transform train/test
```

Therefore, PCA is not fitted on the full dataset before splitting. This is important to state explicitly because reviewers may suspect PCA leakage if the paper only says "we applied PCA" without specifying the order.

Recommended paper wording:

```text
For each split, all preprocessing steps are fitted only on the training subset. In particular, PCA is fitted on the training data and then used to transform the corresponding test data.
```

## PQK: Primary Result

PQK is the primary result because it is stable under the benchmark protocol. For PQK, the selected circuit/configuration is determined from the GA search stage. The repeated benchmark splits are used only to estimate final accuracy after the circuit is frozen.

This supports the following claim:

```text
For GA-PQK, benchmark test labels are not used during GA evolution or post-hoc circuit selection. The benchmark splits are used only for final performance estimation of the frozen selected circuit.
```

In the paper, GA-PQK should be framed as the main quantitative performance claim.

## FQK: Stability Diagnostic

FQK behaves differently from PQK. It is much less stable across candidate circuits and random data splits. Because of this instability, several top candidate circuits/configurations may have been inspected before choosing the best FQK result.

If the best FQK circuit/configuration is selected after looking at benchmark test accuracy, then the FQK number should not be presented as the same kind of unbiased final estimate as PQK. This is not PCA leakage and not GA training directly on the final benchmark test set, but it is post-hoc model selection using benchmark performance.

The clean way to present FQK is:

```text
FQK results are reported as a candidate-sensitivity or best-case diagnostic, not as the primary unbiased performance claim.
```

Recommended paper wording:

```text
Unlike PQK, FQK exhibited substantial sensitivity to the selected evolved circuit and to the random data split. We therefore report GA-FQK as a stability diagnostic. The main quantitative conclusion is based on GA-PQK, whose circuit is fixed before repeated benchmark evaluation.
```

## Mean and Standard Deviation for FQK

The available notebooks contain repeated benchmark results for frozen FQK circuits over seeds 100-109. These can be reported as mean and standard deviation over benchmark splits.

However, this should be described precisely:

```text
The reported standard deviation measures split-level benchmark variability for a frozen evolved circuit. It is not the standard deviation over independent GA searches.
```

If multiple FQK candidate circuits were inspected, the most transparent presentation is a supplementary candidate-sensitivity table:

```text
Candidate circuit | Source folder | Benchmark seeds | Mean accuracy | Std accuracy
```

This avoids hiding the instability and prevents reviewers from interpreting a best-only FQK number as an unbiased estimate.

## What Should Be Changed in the Paper

### Section 4.1: GA-QSVM Procedure

Replace `D_test` in the GA fitness description with `D_val` or `D_search-val`.

Instead of:

```text
QSVM is trained on D_train and its classification accuracy is measured on D_test; this score serves as the fitness.
```

Use:

```text
QSVM is trained on the search-training subset and its classification accuracy is measured on the search-validation subset; this score serves as the GA fitness.
```

Equation (8) should use validation notation, for example `y_val`, not `y_test`.

### Section 5.1: Dataset Preparation

Add the preprocessing order:

```text
For every split, preprocessing is performed after splitting. Scaling and PCA are fitted only on the training subset and then applied to the corresponding test subset.
```

### Section 5.3: GA-QSVM Performance

Add a final benchmark protocol paragraph before Figure 6:

```text
For the final performance comparison, the circuits discovered by GA are frozen before benchmark evaluation. In the GA-PQK setting, the selected circuit is determined from the GA search stage and then evaluated on repeated random train/test splits. The benchmark test labels are not used during GA evolution or circuit selection. For GA-FQK, because FQK shows high sensitivity to the selected evolved circuit, we report it as a stability diagnostic and provide split-level mean and standard deviation.
```

### Figure 6 Caption

Suggested replacement:

```text
Figure 6. Accuracy comparison between classical SVM, standard QSVM kernels, and GA-optimized kernels. For GA-PQK, the circuit is selected during the GA search stage and frozen before repeated benchmark evaluation. For GA-FQK, because FQK exhibits high candidate-circuit sensitivity, the result is reported as a stability diagnostic. Error bars denote standard deviation over repeated benchmark splits. In every split, scaling and PCA are fitted only on the training subset.
```

## Suggested Response to Reviewer

```text
We thank the reviewer for pointing out that the previous description could be interpreted as using the final test set during GA optimization. This was a terminology issue in the manuscript. During GA evolution, the held-out subset used for fitness should be understood as a search-validation subset, not the final benchmark test set. For the main GA-PQK results, the evolved circuit is fixed before final benchmark evaluation, and the benchmark test splits are not used for circuit selection. We have revised the methodology and Figure 6 description to make this protocol explicit.

We also clarify that FQK is substantially less stable than PQK. Therefore, in the revised manuscript GA-FQK is reported as a stability diagnostic with mean and standard deviation over repeated benchmark splits, while the main quantitative claim is based on GA-PQK.
```

## Bottom Line

The defensible position is:

- PCA does not leak because it is fitted only after splitting and only on training data.
- GA-PQK is the primary clean result if the circuit is selected from the GA search stage before benchmark evaluation.
- GA-FQK should be presented as unstable/candidate-sensitive if the best FQK candidate was chosen after inspecting benchmark performance.
- The paper should stop using `test` to describe the GA fitness split and should explicitly describe the two-stage search-then-benchmark protocol.
