# Flexible Genetic Algorithm for Quantum Support Vector Machines

**Authors:** Nguyen Minh Duc\*<sup>1,3</sup>, Vu Tuan Hai<sup>2,3,*</sup>, Le Bin Ho<sup>4,5,†</sup>, and Lan Nguyen Tran<sup>1,3,††</sup>

**Affiliations:**

1. University of Science, Vietnam National University, Ho Chi Minh City 700000, Vietnam.
2. University of Information Technology, Vietnam National University, Ho Chi Minh City 700000, Vietnam.
3. Vietnam National University, Ho Chi Minh City 700000, Vietnam.
4. Frontier Research Institute for Interdisciplinary Sciences, Tohoku University, Sendai 980-8578, Japan.
5. Department of Applied Physics, Graduate School of Engineering, Tohoku University, Sendai 980-8579, Japan.

**E-mail:** \* haivt@uit.edu.vn, † binho@fris.tohoku.ac.jp, †† tnlan@hcmus.edu.vn

**Keywords:** quantum computing, machine learning, quantum support vector machine, genetic algorithms, feature map optimization

## Abstract

Quantum Support Vector Machines (QSVM) is one of the most promising frameworks in quantum machine learning, yet their performance depends on the design of the feature map. Conventional approaches rely on fixed quantum circuits, which often fail to generalize across datasets. To address this limitation, we propose **GA-QSVM**, a hybrid framework that employs Genetic Algorithms (GA) to automatically optimize feature maps. The proposed method introduces a configurable framework that flexibly defines the evolutionary parameters, enabling the construction of adaptive circuits. Experimental evaluation of datasets, including Digits, Fashion, Wine, and Breast Cancer, demonstrates that GA-QSVMs achieve a comparable accuracy compared to classical SVMs and standard QSVMs. Furthermore, transfer learning results indicate that GA-QSVM's circuits generalize effectively across datasets. These findings highlight the potential of evolutionary strategies to automate and enhance kernel design for future quantum machine learning applications.

## 1. Introduction

Quantum Machine Learning (QML) [1, 2] has emerged as a promising field with the potential to offer computational advantages over classical methods. Among various QML algorithms, the Quantum Support Vector Machine (QSVM) [3, 4] is particularly notable, as it translates a well-understood classical Support Vector Machine (SVM) into the quantum domain. For the dual problem, a classical SVM has a complexity of $O(M^2)$ for an $M$-size dataset. In contrast, the quantum dual problem requires $O(M^{4.67}/\epsilon^2)$ quantum circuit evaluations to achieve an $\epsilon$-accurate solution [4]. This indicates that the quantum dual approach grows more rapidly with problem size than the classical method. The power of a QSVM lies within its quantum kernel $K$ between two quantum states $|\psi(x_i)\rangle$ and $|\psi(x_j)\rangle$, which measures the similarity between them. These states are prepared by the quantum feature map $\phi$ through a unitary $U_\phi(\theta)$.

The primary challenge in QSVM is the design of the parameterized circuit used in the feature map. Complex circuits can capture data patterns, but they face two significant issues. First, in high-dimensional feature spaces, the inner product between data vectors $x_i^\top x_j$ concentrates around a constant $\epsilon$, which reduces the variability of the kernel matrix, making it nearly uniform and thus difficult to train. Second, overly expressive circuits often lead to poor generalization, a phenomenon analogous to overfitting in classical machine learning [5, 6]. Currently, QSVM parameterized circuits often rely on fixed ansatzes, such as ZZFeatureMap [7], Hardware Efficient Embedding (HEE) [8, 9] or Instantaneous Quantum Polynomial (IQP)-style circuits [10, 11], which may not be optimal for a general purpose.

To address the challenge of manual circuit design, previous works used Genetic Algorithm (GA) to automatically discover effective circuits [12-18]. GA is a powerful metaheuristic for searching in many quantum problems, such as state preparation [13, 14], tomography [15], image processing [16-18], and quantum compilation [19]. In particular, in the QSVM framework, there have been several works employing GA to search for optimal feature maps. In Ref. [12], the authors applied GA to search gate-sequence encodings, showing that GA-discovered circuits can outperform hand-designed circuits while revealing a positive relationship between classifier test accuracy and kernel entropy. Pello et al. [20] demonstrated that the kernel-target alignment is suitable for the direct accuracy evaluation inside an NSGA-II [21] based GA. Those authors further accelerated the fitness evaluation by using an approximation to kernel-target alignment. Gate parameters are optimized with classical optimizers, producing compact maps with improved alignment. Recently, Wang et al. have studied entanglement explicitly with a multi-objective GA that separates fitness contributions from local and non-local gates [22]. Their work showed that (i) optimal circuits tend to include more non-local gates rather than eliminating them, and (ii) the multi-objective selection yields the Pareto front representing a set of optimal trade-offs where improving accuracy leads to an increase in kernel size.

Existing GA-QSVM approaches still have several limitations. First, their performance often highly depends on the benchmark dataset. As a result, QSVM does not generally outperform classical SVMs on various tasks [12]. Second, previous GA procedures lacked a normalization mechanism to ensure that evolved solutions remain aligned with the original objective [20]. Finally, previous methods lacked well-defined evolutionary strategies to improve fitness in the next generation [22]. To address these limitations, we propose a configurable GA framework that allows the dynamic definition of fitness function, normalizer, generator, and GA operations. Our framework can automatically adjust resources, such as increasing the circuit depth or adding more gates, when GA fails to find an optimal solution. For evaluation, we select four toy datasets that are suitable for simulating QSVM on a classical computer. We then conduct a comparison between classical SVMs, QSVMs with standard feature maps, and our proposed GA-QSVMs. Especially, we also assess the performance of our GA-QSVM with different types of kernels, including Fidelity Quantum Kernel (FQK) and Projected Quantum Kernel (PQK) [23, 24].

The remainder of this paper is organized as follows. Section 2 reviews the related works on QSVM and GA applications in quantum computing. Section 3 presents the background of GA, classical SVM, and quantum kernels. Section 4 describes the proposed GA-QSVM methodology, including the evolutionary design of quantum circuits and metadata configuration. Section 5 provides the evaluation results on benchmark datasets, hyperparameter analysis, and comparison with classical and existing quantum SVMs. Finally, Section 6 concludes the paper and discusses potential future research directions.

## 2. Related works

Recently, QSVM has emerged as one of the prominent QML approaches. Several works [25, 26] provided the foundation for applying quantum-enhanced feature spaces to supervised learning tasks. These early studies demonstrated how quantum feature maps could transform classical data into high-dimensional Hilbert spaces, enabling the classification of non-linear patterns that are difficult to capture with classical kernels. Later works have shown that the QSVM performance depends strongly on the dataset [24, 27]. While QSVM and classical SVM perform comparably for simple datasets, QSVM has been found to outperform the classical one for complex datasets thanks to the better representational of quantum feature maps in these cases. Beyond benchmarking, the QSVM framework has been successfully applied to real-world problems. For example, QSVM can solve the vehicle routing problem [28], air pollution prediction [29], DNA sequence similarities [30], and handling data with underlying group structures through covariant kernel formulations [31].

Despite the above-mentioned advances, some major challenges remain. Mapping classical data into quantum states has proven to be non-trivial, with subtleties that affect both trainability and performance [32]. As the same in classical SVM, overfitting is another recurring issue in QSVM [33, 34]. Furthermore, a key challenge in QSVMs lies in the kernel design. Trainable quantum kernels have been proposed to adapt the kernel based on datasets, improving the ability of overall generalization [35, 36]. Complementing this line of work, Suzuki et al. [37] analyzed the expressivity of quantum feature maps, providing insights into how different kernels affect the performance of classifiers. Gil-Fuster et al. showed that one can improve the performance of QSVM by designing new quantum kernels [38]. Recently, the GA-QSVM approach introduced in Ref. [39] leveraged GA to optimize kernel structure, representing a step toward automating the kernel selection problem. This hybrid evolutionary-quantum strategy enhances the accuracy of QSVMs, particularly when dealing with complex datasets.

## 3. Background

### 3.1. Genetic algorithms

GAs are classes of evolutionary optimization techniques inspired by the biological process of natural selection. They operate by maintaining a population that evolves over generations according to a defined fitness function. A population consists of multiple candidates, forming the search space for the algorithm. The structure of each candidate is encoded as a chromosome serving as its genetic representation. Within each chromosome, a gene corresponds to a specific component of the overall solution [40, 41].

The evolution of the population proceeds by repeating selection, crossover, mutation, and replacement operations. In the selection phase, candidates with higher fitness values are more likely to be chosen as parents for the next generation. The crossover operation then combines parts of two or more parent chromosomes to produce offspring, thereby promoting the exchange of better genes. Mutation introduces small random changes to candidate genes, preserving the genetic diversity and helping the algorithm avoid local minima. Finally, the replacement determines which candidates from the current and new generations will survive to become the next population. Through this iterative process, GA gradually refines candidate solutions, converging toward an optimal or near-optimal result. Their ability to search large, complex, and non-convex spaces without relying on gradient makes them particularly effective in optimization [42-44].

Conventional GAs used in existing GA-QSVM frameworks [12, 20, 22] often face limitations due to their static evolutionary mechanisms, leading to a large deviation from the targeted optimization objective. Earlier GA-QSVM implementations also typically lacked a normalization mechanism to ensure that evolved individuals remain consistent with the targeted fitness landscape, resulting in unstable evolution across generations. Additionally, GA frameworks do not have adaptive strategies to enhance fitness progression effectively in generations. To overcome these issues, recent studies have introduced Adaptive Genetic Algorithms [45, 46] (AGAs) to dynamically adjust their parameters, such as resources, mutation rate, crossover probability, and selection pressure, based on the current population state. These adaptive mechanisms enhance convergence robustness across various environments.

### 3.2. Classical Support Vector Machine

SVM [47, 48] is a popular and effective supervised approach to linear classification. The prediction is made using the function of the inner product of $x$ and a weight vector $w \in X$: $f^*(x)=\operatorname{sign}(f(x)+b)$, where $f(x)=w\cdot x$, defining a separate hyperplane with orientation and offset controlled by $w$ and $b$, respectively.

To handle non-linearly separable data, a soft margin is employed, and a hinge loss is then traded off with the norm of $w$ when training SVM models. We can see it as a regularization part to avoid overfitting. For complex and nonlinear problems, data points are mapped from the input space $X$ to a higher-dimensional space $\tilde{X}$ using a mapping function $\phi: X \to \tilde{X}$. In this new space, a linear decision boundary is found. The training process for a soft-margin SVM is defined by the following primal optimization problem:

$$
\min_{w,b,\epsilon} \frac{1}{2}\|w\|_2^2 + C\sum_{i=1}^{n}\epsilon_i
\tag{1}
$$

subject to:

$$
\epsilon_i \ge 0, \quad y_i(w\cdot \phi(x_i)+b) \ge 1-\epsilon_i \quad \forall i=1,\ldots,n,
\tag{2}
$$

where the parameter $C>0$ determines the trade-off between increasing the margin size and ensuring that the $x_i$ lie on the correct side of the margin.

The kernel trick is used to avoid explicitly computing the mapping $\phi(x_i)$. By using a kernel function, $K(x_i,x_j)$, we can calculate the inner product of vectors in $\tilde{X}$ directly from the original input vectors. This allows the problem to be solved using its dual formulation, expressed as:

$$
\max_\beta L(\beta)=\sum_{i=1}^{n}\beta_i-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\beta_i\beta_j y_i y_j K(x_i,x_j),
\tag{3}
$$

subject to:

$$
\sum_{i=1}^{n}\beta_i y_i=0, \quad 0\le \beta_i\le C \quad \forall i=1,\ldots,n.
\tag{4}
$$

**Algorithm 1. QSVM $F$**

```text
Require: Dataset D, quantum circuit g
1: Construct quantum kernel K with g as feature map
2: Construct {x_train, y_train, x_test, y_test} from D.
3: Train the classifier C_K with {x_train, y_train}
4: y_pred <- C_K(x_test)
5: return Accuracy between {y_test, y_pred}
```

Classical SVMs fundamentally rely on user-chosen kernels whose inductive bias may not capture higher-order, non-local correlations in the data. In practice, performance depends heavily on heuristic kernel selection and bandwidth tuning; poorly tuned kernels either overfit (small bandwidth) or "wash out" structure (large bandwidth), and hyperparameters interact in nontrivial ways [49]. Moreover, kernel methods scale at least quadratically in sample size due to the $N\times N$ Gram matrix, creating time/memory bottlenecks for large $N$ and complicating cross-validation and model selection. Even sophisticated classical kernels or twin-SVM variants cannot fully avoid these scaling limits [50].

### 3.3. Quantum Support Vector Machines

QSVMs are quantum versions of classical SVMs. As defined in Algorithm 1, they leverage quantum principles to handle high-dimensional data. QSVMs define kernels via state fidelity in exponentially large Hilbert spaces using data-encoding circuits. Carefully designed circuits can embed classically hard-to-compute similarities, potentially improving generalization for certain tasks [49]. In QSVM, the datapoint $x_i$ is encoded into a quantum state $|\psi(x_i)\rangle$. This mapping is achieved through a quantum feature map $U(\theta,x)$ applied to an initial state $|0\rangle$:

$$
|\psi(x)\rangle = U(\theta,x)|0\rangle.
\tag{5}
$$

The widely-used QSVM kernel is FQK $K$, defined as the fidelity between two quantum states encoded by the circuit $g$, denoted as $K_g^{FQK}(x_i,x_j)$:

$$
K_g^{FQK}(x_i,x_j)=|\langle\psi(x_i)|\psi(x_j)\rangle|^2.
\tag{6}
$$

Each input $x_i$ is then fed into classifier $C_K$, which returns the corresponding label $y_i$.

To overcome the drawback of standard FQK suffering from small geometric differences when the effective dimension $d$ is large, PQK has been recently developed by Huang et al. [6]. PQK projects quantum states into approximate classical representations, such as reduced physical observables or classical shadows, enabling better generalization in low-dimensional classical spaces. The simple form of PQK measures the one-particle reduced density matrix (1-RDM) for each qubit: $\rho_k(x_i)=\operatorname{Tr}_{j\ne k}[\rho(x_i)]$ and is defined as

$$
K_g^{PQK}(x_i,x_j)=\exp\left(-\gamma\sum_k \|\rho_k(x_i)-\rho_k(x_j)\|_F^2\right),
\tag{7}
$$

which is known as the Frobenius norm between two 1-RDM. PQK can be efficiently computed via classical shadows, providing a practical route for quantum-enhanced learning with fewer measurements. Recent works have shown that quantum kernels are efficient in multiclass classification when quantum circuits and their hyperparameters are chosen properly [8, 51, 52]. In particular, QSVMs can bypass classical SVM limitations when (i) the quantum feature map targets data structures inapplicable to classical kernels, and (ii) hyperparameters are carefully managed [52]. The GA approach with dynamically designing quantum circuits for the kernel is expected to solve the two above-mentioned problems.

**Table 1. GA-QSVM terminology mapping**

| GA | GA-QSVM | Description |
|---|---|---|
| Gene | Quantum gate $g$ | Unitary operator sampled from pool gates |
| Candidate | Parameterized quantum circuit $g$ | A set of $g$ |
| Population | Generation $G^{(i)}$ | A set of $g$ |
| Fitness | Fitness | Performance metric |

![Figure 1. GA-QSVM procedure flowchart.](document_markdown_assets/figure1_ga_qsvm_flowchart.png)

**Figure 1.** The flowchart of the GA-QSVM procedure. At generation $i$, we evaluate set $G$ then update this set until at least one fitness value meets the threshold.

## 4. Methodology

Our approach combines GA and QSVM into a hybrid workflow as a two-level optimizer. QSVM serves as a low-level optimizer to find an optimal hyperplane, whereas GA is the high-level optimizer to find the best circuit, which strongly affects the QSVM optimization.

### 4.1. GA-QSVM procedure

Our GA-QSVM procedure is presented in Figure 1. Each candidate in the population represents a circuit composed of a sequence of gates from a pool. Notice that previous works included RX, RZ, and CX gates restricted to 90-degree ($\sqrt{X}$), but lack of the Y-axis rotation gate [20]. In this work, to enhance the flexibility, we involve $R_x(\theta)$, $R_y(\theta)$, $R_z(\theta)$, H, CX gates.

The corresponding fitness $F(\ldots)$ of each circuit is determined by its performance for a given dataset $D$ that is separated into $\{D_{train}, D_{test}\}$. As given in Algorithm 2, for each circuit in the population, we perform QSVM using the corresponding quantum kernel. QSVM is trained on $D_{train}$ and its classification accuracy is measured on $D_{test}$, this score serves as the fitness:

$$
\mathrm{Accuracy}=\frac{1}{|y_{test}|}\sum_{j=1}^{|y_{test}|} \mathbb{I}(\hat{y}_{test_j}==y_{test_j}),
\tag{8}
$$

where $\mathbb{I}(\cdot)$ is the indicator function, which is 1 if the predicted label $\hat{y}$ matches the true label $y$ and 0 otherwise.

At the end of each generation, the circuits are ranked by their fitness. The evolution then proceeds with three steps: selection, crossover, and mutation as depicted in Figure 2. We employ the elitist selection strategy, in which the best-performing circuits of the population are chosen to be parents for the next generation. Then, it was divided into the father set and the mother set. The number of eliminated circuits is set to $n_{circuit}/2$ by default, meaning that the population is kept over generation if the initial population is divisible by 4.

$$
\{G_{father}^{(i)},G_{mother}^{(i)}\}\leftarrow \mathrm{Selection.Elitist}(G^{(i)},F,n_{circuit}/2).
\tag{9}
$$

These selected circuits undergo crossover to create a new generation of offspring. We stress that, unlike previous GA-QSVM frameworks [20], our approach forces the offspring to be normalized under several conditions that balance the expressivity and trainability of quantum circuits. For example, (1) circuits should be shortened if their depth is longer than a pre-defined depth $d$, and (2) CX gates must be added if the number of CX gates is less than the number of required CX gates.

$$
G^{(i+1)} \leftarrow \mathrm{Crossover.OnePoint}(\{G_{father}^{(i)},G_{mother}^{(i)}\}, \mathrm{Normalizer.Depth}(d),\ldots).
\tag{10}
$$

We use the bit-flip mutation with a mutation rate $p$. For each gate $g$, there is a $p\%$ chance that $g$ changes to a different gate belonging to the pool gates. The value $p$ is thus crucial to find an optimal circuit. If $p$ is low, GA-QSVM will get stuck at local minima; otherwise, the evolution process can be unstable. The mutated circuits also need to be normalized following the normalization condition:

$$
G^{(i+1)} \leftarrow \mathrm{Mutation.BitFlip}(G^{(i+1)}, \mathrm{Normalizer.Depth}(d),\ldots).
\tag{11}
$$

![Figure 2. GA operations on quantum circuits.](document_markdown_assets/figure2_ga_operations.png)

**Figure 2.** GA operations on quantum circuits. (a) Two 6-qubit quantum circuits $\{g_1,g_2\}$ are encoded as genome by circuit depth. (b) Two quantum circuit are crossover: $g_1$ and $g_2$ are divided into $\{g_{1,1}, g_{1,2}, g_{2,1}, g_{2,2}\}$ then $g_{1,1}$ combines with $g_{2,2}$, $g_{2,1}$ combines with $g_{1,2}$. (c) The offspring then randomly mutate. Note that if the offspring are not followed by a normalized condition, the addition/truncation gate's operation may be applied due to the normalization condition.

**Algorithm 2. GA for QSVM task**

```text
Require: Fitness function F, metadata M
1: G^(0) <- Generator(M)
2: for i in [0 ... n_generation] do
3:     F <- [F(g) for g in G^(i)]
4:     if exists f in F, f < M_tau then
5:         return G^(i)
6:     end if
7:     {G_father^(i), G_mother^(i)} <- Selection(G^(i), Metadata)
8:     G^(i+1) <- Mutation(Crossover({G_father^(i), G_mother^(i)}, ...), ...)
9: end for
10: return G^(i)
```

### 4.2. Metadata for GA-QSVM

In the proposed GA-QSVM framework, metadata is defined as a tuple of hyperparameters that configure both the quantum feature map and the GA process. Formally, we denote the metadata as

$$
M=(\tau,n,n_{R_x},n_{R_y},n_{R_z},d,n_{circuits},n_{generation},p_m),
\tag{12}
$$

where $\tau$ is the threshold, $n$ is the number of qubits, $(n_{R_x},n_{R_y},n_{R_z})$ represent the allocation of single-qubit rotation gates along the $x$, $y$, and $z$ axes, $d$ denotes the circuit depth, $n_{circuits}$ is the number of circuits in the population, $n_{generation}$ is the number of generations, and $p_m$ is the mutation probability. The first subset of parameters $(n,n_{R_x},n_{R_y},n_{R_z},d)$ governs the expressibility of the quantum circuit, while the second subset $(n_{circuits},n_{generation},p_m)$ regulates the behavior of the genetic algorithm by controlling the balance between exploration and exploitation. By encapsulating these parameters in $M$, the framework ensures reproducibility, enables systematic benchmarking across hyperparameter configurations.

**Table 2. Dataset properties**

| Dataset | Ref | #Instances | #Classes | #Features |
|---|---:|---:|---:|---:|
| Digits | [57] | 5620 | 10 | 64 |
| Fashion | [58] | 70000 | 10 | 784 |
| Wine | [59] | 178 | 3 | 13 |
| Breast Cancer | [60] | 592 | 2 | 30 |

![Figure 3. Cumulative explained variance after using PCA.](document_markdown_assets/figure3_pca_explained_variance.png)

**Figure 3.** Cumulative explained variance of (a) Digits, (b) Fashion, (c) Wine, and (d) Breast Cancer dataset after using PCA. The number of components for 95% cumulative explained variance for four datasets is 30, 200, 10, and 10.

## 5. Numerical results

### 5.1. Dataset preparation

We have performed QSVM with two types of quantum kernels, including PQK and FQK using Qiskit 1.3.1 [53] and Squlearn 0.8.4 [54], respectively. Regarding datasets, we used Scikit-learn 1.6.1 [55] for wine, digits, and breast-cancer datasets and Tensorflow 2.16.2 [56] for the Fashion MNIST dataset. All calculations were performed on an Intel Core i9-10920X CPU at 3.50 GHz and 120 GB of RAM. For comparison, we have used a radial basis (RBF) kernel for classical SVM, and Z/ZZ feature map with FQK and PQK for conventional QSVM.

We use four datasets for the classification task, including: Digits, Fashion, Wine, and Breast Cancer as described in Table 2. The maximum number of training items and testing items is 100 and 100 for all experiments, respectively. Only three datasets, including Digits, Wine, and Breast Cancer were used in GA-QSVM because of the huge number of Instances (#Instances) from Fashion. Due to the limitation of the number of qubits, Principal Component Analysis (PCA) was applied to reduce the number of Features (#Features) of data points $\{x_i\}$. Note that the number of qubits must be equal to #Feature because we use the angle encoding method. Some first experiments are shown in Figure 3, the number of principal components needed to retain a significant amount of information varies greatly by dataset. The Wine and Breast Cancer datasets are less complex, requiring only 10 components for 95% information, while the Digits and Fashion datasets, especially the latter, require a much higher number of components.

### 5.2. Optimal circuit searching

As standard GA, we measure the best fitness versus generation. As an example, we plot the searching process for the digit dataset in Figure 4. The best fitness starts from a low value, becomes better through generations, and reaches an upper bound limit that comes from the resources we restrict and the dataset. We investigate how hyperparameters affect the best fitness. The benchmark results show the effect of different configurations: $n_{CX}\in[5,10,15,20,25]$, $p\in[0.001,0.01,0.1,0.3,0.5]$, $d\in[5,10,15,20,25]$, $n_{circuit}\in[4,8,16,20]$.

![Figure 4. Hyperparameter survey on the Digits dataset.](document_markdown_assets/figure4_hyperparameter_survey.png)

**Figure 4.** We conduct the survey on hyperparameters by benchmarking on Digits dataset, with fixed $n=5$, $n_{circuit}=16$, $p=0.1$, $d=5n$, $nCX=2n$ and different configuration. (a) different $d$, (b) different $n_{circuit}$, (c) different $nCX$ and (d) different $p$. Note that in GA-QSVM, we run the fitness function (QSVM) with fewer iterations to save the whole optimization time.

As shown in Figure 4a, the circuit depth $d$ has a noticeable effect on the performance; deeper circuits, such as $d=20$ and $d=25$, lead to a higher and more stable fitness. The size of the population $n_{circuit}$, shown in Figure 4b, has a significant impact on performance and convergence. Larger populations, such as $n_{circuit}=16$ and $n_{circuit}=20$, converge more quickly and achieve a higher, more stable final fitness with less variance. As we can see in Figure 4c and d, the best fitness for different $n_{CX}$ and $p$ values are tightly close to each other with the peak fitness around 0.79. Within the tested range, the number of CX gates and mutation probability do not significantly impact the final classification accuracy. Overall, as the depth and the population size increase, more resources are included in the GA process, leading to better circuit searching.

![Figure 5. Optimal circuits selected by GA-QSVM.](document_markdown_assets/figure5_optimal_circuits.png)

**Figure 5.** Optimal 7-qubit circuits from GA-QSVM for the three datasets (a) Digits, (b) Wine, and (c) Breast Cancer with two quantum kernels (1) FQK and (2) PQK.

Figure 5 illustrates the optimal 7-qubit circuits selected by the GA-QSVM framework. Each subfigure (a)-(c) corresponds respectively to the Digits, Wine, and Breast Cancer datasets, while rows (1) and (2) show the best circuits produced under the FQK and PQK kernels. For all datasets, GA yields shallow circuits combining $\{R_x,R_y,R_z,H,CX\}$. Notably, the FQK-optimized circuits tend to add more CX gates, whereas PQK-optimized circuits often emphasize balance between local and non-local gates. The structural differences across datasets highlight how the GA adapts circuit topology to dataset complexity: the Breast Cancer datasets favor shallower constructions with fewer CX gates.

### 5.3. GA-QSVM performance

![Figure 6. Accuracy comparison between classical and quantum SVM models.](document_markdown_assets/figure6_accuracy_comparison.png)

**Figure 6.** Accuracy between models classical SVM (RPF), existing QSVM kernel (FQK, PQK) and our optimized kernel (GA-FQK, GA-PQK) with default configuration $n_{circuit}=16$, $p=0.1$, $d=5$, $nCX=2n$; which use (1) standard train/test/val split (from 4 to 7 qubits) and (2) k-fold (from 3 to 7 qubits). The QSVM models with optimized kernel are run with full number of iterations.

Figure 6 compares the accuracy of different methods (GA-FQK, GA-PQK, FQK, PQK, RBF) for three datasets using the standard train/test/val split (subfigure (1)) and k-fold cross-validation (subfigure (2)). In k-fold cross-validation, the QSVM is trained $k$ times through $k$ train-test splits, then the results are averaged to obtain the final accuracy [61]. The performance of the standard train/test/val method is not stable because it depends on how the test set is split. In contrast, k-fold cross-validation yields a more robust estimate of the model's generalization performance [62].

We can see that the conventional QSVM (FQK/PQK) with a fixed ansatz shows the worst accuracy for all cases. For the standard train/test/val method, the performance of GA-FQK/GA-PQK and RBF is comparable. However, the performance of GA-QSVM with both FQK and PQK is noticeably better than the classical kernel RBF for the k-fold cross-validation (subfigure (2)). In general, the comparison highlights that our proposed methods maintain competitive or superior performance under the standard evaluation, and both quantum and classical SVMs achieve near-saturation accuracy.

To prove the expressivity of GA-QSVM, we run GA-QSVM on a base dataset, then use the solution (the best circuit) as an ansatz for QSVM on other datasets. As an example, we choose Digits as the source for transfer because it is a complex dataset, forcing the GA-QSVM to learn a highly expressive ansatz. This expressive structure is then expected to transfer to other datasets. Transfer learning aims to reduce the number of trials for GA-QSVM on other datasets.

![Figure 7. Transfer learning accuracies.](document_markdown_assets/figure7_transfer_learning.png)

**Figure 7.** Accuracies from our optimized kernels on different datasets using transfer learning (right columns). The number of features (number of qubits) used in this experiment is 10.

Figure 7 summarizes the performance of transfer learning across different target datasets when using Digits as the base dataset. To enhance the expressivity of the transferred circuit, we have used 10 features corresponding to 10 qubits. The first two columns are results from the GA-QSVM original training for Digits and Fashion, whereas the last three columns represent the accuracy of QSVM for Fashion, Wine, and Breast Cancer using the ansatz transferred from Digits.

Fashion is a complex dataset that the classical SVM can only achieve an accuracy of 65.7%, and QSVM/PQK with the standard Z-Feature Map (with $n=10$) has a very low accuracy of 22.9%. Also, we can see that the Fashion dataset is so complex that the original GA-QSVM cannot achieve high accuracies (only 50.2% and 46.5% for FQK and PQK, respectively). However, when the ansatz transferred from Digits is used for Fashion, the accuracy is significantly improved. Especially, the FQK accuracy is up to 68.1%, higher than the classical SVM (65.7%). This improvement may arise from the ansatz learned on Digits, capturing richer feature correlations and offering a more favorable inductive bias than training on Fashion alone. For Wine and Breast Cancer, the transfer learning can also achieve reasonable accuracies. The comparison also reveals that FQK overall outperforms PQK in transfer learning, except for the Breast Cancer where PQK shows superior performance (85.7 vs 79.1). In general, using transfer learning can help to reduce the computational demands of GA training. Still, an open question is how to choose a proper base dataset to get an expressive ansatz for various datasets.

## 6. Conclusion

By allowing flexible metadata configuration and adaptive search, GA-QSVM overcomes the limitations of manually designed circuits and improves kernel expressivity. Numerical results on various datasets demonstrate that GA-QSVM achieves an accuracy superior to standard QSVM with a fixed quantum feature map, especially for complex data. Moreover, the transfer-learning evaluation highlights that a GA-QSVM circuit trained on one dataset can be effectively generalized across domains, thereby reducing training time for different datasets.

The GA-QSVM approach also presents limitations. The optimization process is still computationally expensive due to repeated kernel evaluations, and the stochastic nature of GA can lead to instabilities across runs. Additionally, as the number of qubits increases, circuit search and evaluation become much more difficult, restricting the scalability on near-term devices. Future work will focus on extending GA-QSVM to multi-objective optimization, enabling simultaneous control over accuracy, circuit depth, and entanglement cost. These directions may further enhance the practicality of evolutionary quantum kernel design for real-world machine learning tasks.

## Acknowledgment

This research is funded by the National Foundation for Science and Technology Development (NAFOSTED) Grant Number 103.01-2024.06. L.B.H. is supported by JSPS KAKENHI Grant Number 23K13025 and the Tohoku Initiative for Fostering Global Researchers for Interdisciplinary Sciences (TI-FRIS) of MEXT's Strategic Professional Development Program for Young Researchers.

## Data and code availability

Data are available from the corresponding authors upon reasonable request. The code is available at <https://github.com/vutuanhai237/GA-QSVM>.

## References

- [1] M. Schuld and N. Killoran, “Quantum machine learning in feature hilbert spaces,” Phys. Rev. Lett., vol. 122, p. 040504, Feb 2019.
- [2] M. Schuld, I. Sinayskiy, and F. Petruccione, “An introduction to quantum machine learning,” Contemporary Physics, vol. 56, no. 2, pp. 172–185, 2015.
- [3] P. Rebentrost, M. Mohseni, and S. Lloyd, “Quantum support vector machine for big data classification,” Phys. Rev. Lett., vol. 113, p. 130503, Sep 2014.
- [4] G. Gentinetta, A. Thomsen, D. Sutter, and S. Woerner, “The complexity of quantum support vector machines,” Quantum, vol. 8, p. 1225, Jan. 2024.
- [5] V. Havlı́ček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, “Supervised learning with quantum-enhanced feature spaces,” Nature, vol. 567, no. 7747, pp. 209–212, Mar 2019.
- [6] H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, and J. R. McClean, “Power of data in quantum machine learning,” Nature Communications, vol. 12, no. 1, p. 2631, May 2021.
- [7] M. Schuld, A. Bocharov, K. M. Svore, and N. Wiebe, “Circuit-centric quantum classifiers,” Phys. Rev. A, vol. 101, p. 032308, Mar 2020.
- [8] S. Thanasilp, S. Wang, M. Cerezo, and Z. Holmes, “Exponential concentration in quantum kernel methods,” Nature Communications, vol. 15, no. 1, p. 5200, Jun 2024.
- [9] J. Leng, J. Li, Y. Peng, and X. Wu, “Expanding Hardware-Efficiently Manipulable Hilbert Space via Hamiltonian Embedding,” Quantum, vol. 9, p. 1857, Sep. 2025.
- [10] E. Recio-Armengol and J. Bowles, “Iqpopt: Fast optimization of instantaneous quantum polynomial circuits in jax,” arXiv preprint arXiv:2501.04776, 2025.
- [11] D. Hangleiter, M. Kalinowski, D. Bluvstein, M. Cain, N. Maskara, X. Gao, A. Kubica, M. D. Lukin, and M. J. Gullans, “Fault-tolerant compiling of classically hard instantaneous quantum polynomial circuits on hypercubes,” PRX Quantum, vol. 6, p. 020338, May 2025.
- [12] F. M. Creevey, J. A. Heredge, M. E. Sevior, and L. C. Hollenberg, “Kernel alignment for quantum support vector machines using genetic algorithms,” arXiv preprint arXiv:2312.01562, 2023.
- [13] F. M. Creevey, C. D. Hill, and L. C. L. Hollenberg, “Gasp: a genetic algorithm for state preparation on quantum computers,” Scientific Reports, vol. 13, no. 1, p. 11956, Jul 2023.
- [14] S. Fomichev, K. Hejazi, M. S. Zini, M. Kiser, J. Fraxanet, P. A. M. Casares, A. Delgado, J. Huh, A.-C. Voigt, J. E. Mueller, and J. M. Arrazola, “Initial state preparation for quantum chemistry on quantum computers,” PRX Quantum, vol. 5, p. 040339, Dec 2024.
- [15] H. L. D. Linh, V. T. Hai, and L. B. Ho, “Advancing quantum process tomography through universal compilation,” 2025.
- [16] S. Altares-López, J. J. Garcı́a-Ripoll, and A. Ribeiro, “Autoqml: Automatic generation and training of robust quantum-inspired classifiers by using evolutionary algorithms on grayscale images,” Expert Systems with Applications, vol. 244, p. 122984, 2024.
- [17] F. Yan, H. Huang, W. Pedrycz, and K. Hirota, “Review of medical image processing using quantum-enabled algorithms,” Artificial Intelligence Review, vol. 57, no. 11, p. 300, Sep 2024.
- [18] A. Senokosov, A. Sedykh, A. Sagingalieva, B. Kyriacou, and A. Melnikov, “Quantum machine learning for image classification,” Machine Learning: Science and Technology, vol. 5, no. 1, p. 015040, mar 2024.
- [19] V. Tuan Hai, N. Tan Viet, J. Urbaneja, N. Vu Linh, L. Nguyen Tran, and L. Bin Ho, “Multi-target quantum compilation algorithm,” Machine Learning: Science and Technology, vol. 5, no. 4, p. 045057, dec 2024.
- [20] R. Pellow-Jarman, A. Pillay, I. Sinayskiy, and F. Petruccione, “Hybrid genetic optimization for quantum feature map design,” Quantum Machine Intelligence, vol. 6, no. 2, p. 45, Jul 2024.
- [21] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: Nsga-ii,” IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182–197, 2002.
- [22] H. Wang, “Several fitness functions and entanglement gates in quantum kernel generation,” Quantum Machine Intelligence, vol. 7, no. 1, p. 7, Jan 2025.
- [23] R. Martı́nez-Peña, M. C. Soriano, and R. Zambrini, “Quantum fidelity kernel with a trapped-ion simulation platform,” Phys. Rev. A, vol. 109, p. 042612, Apr 2024.
- [24] H.-Y. Huang, M. Broughton, M. Mohseni, R. Babbush, S. Boixo, H. Neven, and J. R. McClean, “Power of data in quantum machine learning,” Nature Communications, vol. 12, no. 1, p. 2631, 2021.
- [25] P. Rebentrost, M. Mohseni, and S. Lloyd, “Quantum support vector machine for big data classification,” Physical Review Letters, vol. 113, no. 13, p. 130503, 2014.
- [26] V. Havlicek, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, “Supervised learning with quantum enhanced feature spaces,” Nature, vol. 567, no. 7747, pp. 209–212, 2019.
- [27] A. Babu, S. G. Ghatnekar, A. Saxena, and D. Mandal, “Entanglement-enabled quantum kernels for enhanced feature mapping,” APL Quantum, vol. 2, no. 1, p. 016116, 2025.
- [28] N. Mohanty, B. K. Behera, and C. Ferrie, “Solving the vehicle routing problem via quantum support vector machines,” Quantum Machine Intelligence, vol. 6, no. 1, p. 34, 2024.
- [29] O. Farooq, M. Shahid, S. Arshad, A. Altaf, F. Iqbal, Y. A. M. Vera, M. A. L. Flores, and I. Ashraf, “An enhanced approach for predicting air pollution using quantum support vector machine,” Scientific Reports, vol. 14, no. 1, p. 19521, 2024.
- [30] C. Shi, G. Leoni, M. Petrillo, A. P. Gallardo, and H. Wang. Compare Similarities Between DNA Sequences Using Permutation-Invariant Quantum Kernel.
- [31] J. R. Glick, T. P. Gujarati, A. D. Corcoles, Y. Kim, A. Kandala, J. M. Gambetta, and K. Temme, “Covariant quantum kernels for data with group structure,” Nature Physics, vol. 20, no. 3, pp. 479–483, 2024.
- [32] S. Thanasilp, S. Wang, N. A. Nghiem, P. J. Coles, and M. Cerezo, “Subtleties in the trainability of quantum machine learning models,” Quantum Machine Intelligence, vol. 5, no. 1, p. 21, 2023.
- [33] E. Peters and M. Schuld, “Generalization despite overfitting in quantum machine learning models,” Quantum, vol. 7, p. 1210, 2023.
- [34] S. Jerbi, L. J. Fiderer, H. Poulsen Nautrup, J. M. Kübler, H. J. Briegel, and V. Dunjko, “Quantum machine learning beyond kernel methods,” Nature Communications, vol. 14, no. 1, p. 517, 2023.
- [35] T. Hubregtsen, D. Wierichs, E. Gil-Fuster, P.-J. H. S. Derks, P. K. Faehrmann, and J. J. Meyer, “Training Quantum Embedding Kernels on Near-Term Quantum Computers,” Physical Review A, vol. 106, no. 4, p. 042431, 2022.
- [36] L. Xu, X.-y. Zhang, M. Li, and S.-q. Shen. Quantum Classifiers with Trainable Kernel.
- [37] Y. Suzuki, H. Yano, Q. Gao, S. Uno, T. Tanaka, M. Akiyama, and N. Yamamoto, “Analysis and synthesis of feature map for kernel-based quantum classifier,” Quantum Machine Intelligence, vol. 2, no. 1, p. 9, 2020.
- [38] E. Gil-Fuster, J. Eisert, and V. Dunjko, “On the expressivity of embedding quantum kernels,” Machine Learning: Science and Technology, vol. 5, no. 2, p. 025003, 2024.
- [39] F. M. Creevey, J. A. Heredge, M. E. Sevior, and L. C. L. Hollenberg. Kernel Alignment for Quantum Support Vector Machines Using Genetic Algorithms.
- [40] G. Acampora, A. Chiatto, and A. Vitiello, “Genetic algorithms as classical optimizer for the quantum approximate optimization algorithm,” Applied Soft Computing, vol. 142, p. 110296, 2023.
- [41] B. Alhijawi and A. Awajan, “Genetic algorithms: theory, genetic operators, solutions, and applications,” Evolutionary Intelligence, vol. 17, no. 3, pp. 1245–1256, Jun 2024.
- [42] Y. Dong and J. Zhang, “An improved hybrid quantum optimization algorithm for solving nonlinear equations,” Quantum Information Processing, vol. 20, no. 4, p. 134, Apr 2021.
- [43] F. Wang, K. Xie, L. Han, M. Han, and Z. Wang, “Research on support vector machine optimization based on improved quantum genetic algorithm,” Quantum Information Processing, vol. 22, no. 10, p. 380, Oct 2023.
- [44] Y. Zhang, J. Zhao, Y. Jia, and X. Shen, “An improved adaptive quantum genetic algorithm as classical optimizer for the quantum approximate optimization algorithm on maxcut problem,” Quantum Information Processing, vol. 24, no. 7, p. 222, Jul 2025.
- [45] M. Gen and L. Lin, Genetic Algorithms and Their Applications. London: Springer London, 2023, pp. 635–674.
- [46] J. Li, R. Liu, and R. Wang, “Handling dynamic capacitated vehicle routing problems based on adaptive genetic algorithm with elastic strategy,” Swarm and Evolutionary Computation, vol. 86, p. 101529, 2024.
- [47] M. Hearst, S. Dumais, E. Osuna, J. Platt, and B. Scholkopf, “Support vector machines,” IEEE Intelligent Systems and their Applications, vol. 13, no. 4, pp. 18–28, 1998.
- [48] W. S. Noble, “What is a support vector machine?” Nature Biotechnology, vol. 24, no. 12, pp. 1565–1567, Dec 2006.
- [49] M. Incudini, F. Martini, and A. D. Pierro, “Toward useful quantum kernels,” Advanced Quantum Technologies, vol. n/a, no. n/a, p. 2300298, 2023.
- [50] S. Egginger, A. Sakhnenko, and J. M. Lorenz, “A hyperparameter study for quantum kernel methods,” Quantum Machine Intelligence, vol. 6, no. 2, p. 44, Jul 2024.
- [51] C. Ding, S. Wang, Y. Wang, and W. Gao, “Quantum machine learning for multiclass classification beyond kernel methods,” Phys. Rev. A, vol. 111, p. 062410, Jun 2025.
- [52] J. Schnabel and M. Roth, “Quantum kernel methods under scrutiny: a benchmarking study,” Quantum Machine Intelligence, vol. 7, no. 1, p. 58, Apr 2025.
- [53] A. Javadi-Abhari, M. Treinish, K. Krsulich, C. J. Wood, J. Lishman, J. Gacon, S. Martiel, P. D. Nation, L. S. Bishop, A. W. Cross et al., “Quantum computing with qiskit,” arXiv preprint arXiv:2405.08810, 2024.
- [54] D. A. Kreplin, M. Willmann, J. Schnabel, F. Rapp, M. Hagelüken, and M. Roth, “squlearn: A python library for quantum machine learning [focus: Quantum software and its engineering],” IEEE Software, vol. 42, no. 5, pp. 65–72, 2025.
- [55] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
- [56] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard, M. Kudlur, J. Levenberg, R. Monga, S. Moore, D. G. Murray, B. Steiner, P. Tucker, V. Vasudevan, P. Warden, M. Wicke, Y. Yu, and X. Zheng, “TensorFlow: A system for Large-Scale machine learning,” in 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16). Savannah, GA: USENIX Association, Nov. 2016, pp. 265–283.
- [57] C. Kaynak, “Methods of combining multiple classifiers and their applications to handwritten digit recognition,” Unpublished master’s thesis, Bogazici University, 1995.
- [58] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms,” arXiv preprint arXiv:1708.07747, 2017.
- [59] S. Aeberhard and M. Forina, “Wine,” UCI Machine Learning Repository, 1992, DOI: https://doi.org/10.24432/C5PC7J.
- [60] W. Wolberg, O. Mangasarian, N. Street, and W. Street, “Breast Cancer Wisconsin (Diagnostic),” UCI Machine Learning Repository, 1993, DOI: https://doi.org/10.24432/C5DW2B.
- [61] T. G. Dietterich, “Approximate statistical tests for comparing supervised classification learning algorithms,” Neural Computation, vol. 10, no. 7, pp. 1895–1923, 10 1998.
- [62] A. B. Manual, “An introduction to statistical learning with applications in r,” 2013.
