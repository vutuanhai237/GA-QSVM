# GA-QSVM — Kế hoạch phát triển codebase theo Review MLST-104723

> **Mục đích tài liệu:** Roadmap kỹ thuật chi tiết để xử lý từng bình luận của reviewer, áp dụng trực tiếp vào repo `vutuanhai237/GA-QSVM`. Có thể dùng làm tài liệu đọc, hoặc làm context cho AI (Claude/Copilot/Cursor) khi viết code.

---

## ⚠️ DEADLINE: 01-JUN-2026

- **Hôm nay:** 19/05/2026
- **Còn lại:** ~13 ngày
- **Editor đã ghi rõ:** "Please let us know if you need more time" → **mạnh tay xin extension**
- **Decision type:** Major revision (DEC:ModRev:S) — 2 referees

**Hai phương án:**
1. **Xin extension** (recommended) — xem template email ở Section 7 cuối file
2. **Crash plan 13 ngày** — xem Section 4-bis bên dưới, chỉ làm critical items

---

## 0. Context cho AI (đọc trước khi sinh code)

**Bài báo:** "Flexible Genetic Algorithm for Quantum Support Vector Machines" — Nguyen et al., submitted to *Machine Learning: Science and Technology* (MLST-104723).

**Decision:** Major revision với 2 referees:
- **Referee 1** (file MLST-104723_review.pdf, ngày 25/04/2026): 7 major + 7 minor comments
- **Referee 2** (trong decision letter ngày 19/05/2026): 4 suggestion points

**Tech stack hiện tại:**
- `Qiskit 1.3.1` — quantum circuits & FQK
- `Squlearn 0.8.4` — PQK kernels
- `Scikit-learn 1.6.1` — classical SVM, datasets, metrics
- `TensorFlow 2.16.2` — Fashion MNIST dataset loader
- Hardware: Intel Core i9-10920X, 120 GB RAM (CPU simulation only)

**Repo:** https://github.com/vutuanhai237/GA-QSVM

**Datasets:** Digits, Fashion-MNIST, Wine, Breast Cancer (Wisconsin) — đều dùng max 100 train + 100 test.

**Core components hiện có:**
- GA loop với operators: Selection (elitist), Crossover (one-point), Mutation (bit-flip)
- Gate pool: `{Rx(θ), Ry(θ), Rz(θ), H, CX}`
- Normalizer cho depth và CX count
- Metadata tuple `M = (τ, n, n_Rx, n_Ry, n_Rz, d, n_circuits, n_generation, p_m)`
- Kernels: FQK (Qiskit) và PQK (Squlearn, dùng 1-RDM Frobenius)

**Mục tiêu tổng:** Biến bản submission hiện tại từ "major revision needed" → "ready for resubmission" thông qua các nâng cấp ở mức **codebase**, **experiments**, và **manuscript**.

---

## 1. Cấu trúc repo đề xuất sau khi nâng cấp

```
GA-QSVM/
├── README.md                          # cập nhật badges, citation, hardware notes
├── requirements.txt                   # pin versions; thêm qiskit-aer, qiskit-ibm-runtime
├── pyproject.toml                     # (mới) cho reproducibility
├── data/
│   └── prepared/                      # cached PCA-reduced datasets (npz)
├── src/
│   ├── ga_qsvm/
│   │   ├── __init__.py
│   │   ├── ga/                        # core GA components (đã có, refactor)
│   │   │   ├── selection.py
│   │   │   ├── crossover.py
│   │   │   ├── mutation.py
│   │   │   ├── normalizer.py
│   │   │   └── metadata.py
│   │   ├── fitness/
│   │   │   ├── qsvm_fitness.py        # MỚI: nested CV-aware fitness
│   │   │   └── leakage_guard.py       # MỚI: enforce val/test separation
│   │   ├── kernels/
│   │   │   ├── fqk.py
│   │   │   ├── pqk.py
│   │   │   └── noise_aware.py         # MỚI: wrap kernel với noise model
│   │   ├── baselines/                 # MỚI toàn bộ
│   │   │   ├── random_search.py
│   │   │   ├── greedy_search.py
│   │   │   └── fixed_ansatz.py
│   │   ├── transfer/                  # MỚI: expanded TL
│   │   │   ├── source_selection.py
│   │   │   └── circuit_analysis.py
│   │   ├── stats/                     # MỚI
│   │   │   ├── multirun.py
│   │   │   └── significance.py
│   │   └── utils/
│   │       ├── timing.py              # MỚI: wall-clock tracker
│   │       └── circuit_counter.py     # MỚI: count circuit evals
├── experiments/
│   ├── exp01_noise_simulation.py      # MỚI
│   ├── exp02_hardware_ibmq.py         # MỚI (optional)
│   ├── exp03_statistical_runs.py      # MỚI
│   ├── exp04_baselines_comparison.py  # MỚI
│   ├── exp05_extended_transfer.py     # MỚI
│   ├── exp06_compute_cost.py          # MỚI
│   └── exp07_nested_evaluation.py     # MỚI
├── notebooks/
│   └── figures_regeneration.ipynb     # regenerate fig 4-7 với error bars
├── results/
│   ├── raw/                           # JSON outputs per run
│   ├── aggregated/                    # mean/std CSV
│   └── figures/                       # final paper figures
├── tests/                             # MỚI: unit tests cho components
└── docs/
    ├── revision-response.md           # point-by-point response
    └── reproducibility.md             # how-to reproduce
```

---

## 2. Bảng ưu tiên tổng hợp

**Note:** Các task ID bắt đầu bằng `R1-Tx` là từ Referee 1, `R2-Tx` từ Referee 2. Task ID gốc `Tx` không đổi để tránh phá link.

| ID | Task | Reviewer comment | Mức độ | Effort (ngày) | Bắt buộc? |
|---|---|---|---|---|---|
| T1 | Sửa test-set leakage (nested eval) | R1 Major #4 | 🔴 Critical | 2–3 | ✅ Yes |
| T2 | Statistical analysis (multi-run + std) | R1 Major #3 + R2 #4 | 🔴 Critical | 3–4 | ✅ Yes |
| T3 | Thêm citations + viết lại Section 2 | R1 Major #7 | 🟡 High | 0.5 | ✅ Yes |
| T4 | Noise-aware simulation | R1 Major #1a | 🔴 Critical | 4–5 | ✅ Yes |
| T5 | Computational cost tracking | R1 Major #2 + R2 #3 | 🟡 High | 1–2 | ✅ Yes |
| T6 | Baseline comparison (random/greedy) | R1 Major #5 | 🟡 High | 3–4 | ✅ Yes |
| T7 | Mở rộng transfer learning + complexity classification | R1 Major #6 + **R2 #1** | 🟡 High | 3–5 hoặc 0.5 | ✅ Yes |
| T8 | Real hardware execution (IBM/IonQ) | R1 Major #1b | 🟢 Optional bonus | 2–3 | ⚠️ Nếu có quota |
| T9 | Backend constraints discussion | R1 Major #1c | 🟡 Medium | 0.5 | ✅ Yes |
| T10 | Hyperparameter summary table | R1 Minor #4 + R2 #4 | 🟢 Low | 0.5 | ✅ Yes |
| T11 | Discuss negative results (nCX, p_m) | R1 Minor #5 | 🟢 Low | 0.5 | ✅ Yes |
| T12 | Moderate "generalizes effectively" claim | R1 Minor #6 | 🟢 Low | 0.25 | ✅ Yes |
| T13 | Giải thích lý do thật loại Fashion khỏi GA-QSVM | R1 Minor #2 | 🟢 Low | 0.25 | ✅ Yes |
| T14 | Discuss sample size implications | R1 Minor #1 | 🟢 Low | 0.5 | ✅ Yes |
| T15 | Clarify theoretical O(M^4.67/ε²) realization | R1 Minor #3 | 🟢 Low | 0.5 | ✅ Yes |
| T16 | Proofread + fix typos | R1 Minor #7 | 🟢 Low | 1 | ✅ Yes |
| **T17** | **Qiskit fixed-ansatz baselines (PauliFeatureMap, EfficientSU2, TwoLocal)** | **R2 #2** | 🟡 **High** | **2–3** | ✅ **Yes** |
| **T18** | **PCA leakage audit (fit chỉ trên train?)** | **R2 #4** | 🔴 **Critical** | **0.5–1** | ✅ **Yes** |
| **T19** | **Reproducibility document (seeds, GA init, splits)** | **R2 #4** | 🟡 **High** | **1** | ✅ **Yes** |

**Tổng effort ước tính cập nhật:** ~25–34 ngày work → **không thể** xong trong 13 ngày → **xin extension là gần như bắt buộc**.

---

## 3. Chi tiết từng task

### 🔴 T1 — Sửa test-set leakage (Nested evaluation protocol)

**Liên kết:** Major comment #4

**Vấn đề:** Hiện tại fitness của GA = accuracy trên test set cố định, cùng test set được tái sử dụng qua mọi generation → GA overfit topology vào split đó.

**Mục tiêu cụ thể:**
- Tách dữ liệu thành 3 phần: `train` (GA training), `val` (GA fitness), `test` (held-out, chỉ dùng 1 lần cuối)
- Hoặc: average fitness qua K independent splits ở mỗi generation
- Báo cáo final accuracy CHỈ trên held-out test set

**Files cần thay đổi/tạo:**
- `src/ga_qsvm/fitness/qsvm_fitness.py` (refactor)
- `src/ga_qsvm/fitness/leakage_guard.py` (mới — pattern guard)
- `experiments/exp07_nested_evaluation.py` (mới)

**Implementation hint:**

```python
# src/ga_qsvm/fitness/qsvm_fitness.py
from sklearn.model_selection import StratifiedShuffleSplit

class NestedFitnessEvaluator:
    """
    Tránh test leakage bằng cách dùng val set cho fitness, test set chỉ touch 1 lần.
    """
    def __init__(self, X, y, test_size=0.2, val_size=0.2, n_inner_splits=3, seed=42):
        # 1. Split ra held-out test trước (không bao giờ touch trong GA)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_trainval, idx_test = next(sss.split(X, y))
        self.X_trainval, self.y_trainval = X[idx_trainval], y[idx_trainval]
        self.X_test, self.y_test = X[idx_test], y[idx_test]
        # 2. Setup K inner splits trên trainval cho fitness eval
        self.inner_cv = StratifiedShuffleSplit(
            n_splits=n_inner_splits, test_size=val_size, random_state=seed
        )

    def fitness(self, circuit, kernel_type='FQK'):
        """Average accuracy qua n_inner_splits val splits."""
        scores = []
        for idx_tr, idx_val in self.inner_cv.split(self.X_trainval, self.y_trainval):
            # train QSVM trên fold train, score trên fold val
            scores.append(self._score_one_split(
                circuit, idx_tr, idx_val, kernel_type
            ))
        return np.mean(scores)

    def final_test(self, best_circuit, kernel_type='FQK'):
        """Gọi đúng 1 lần sau khi GA xong."""
        return self._score_one_split_full_test(best_circuit, kernel_type)
```

**Acceptance criteria:**
- [ ] Test set không xuất hiện trong vòng fitness của GA (assertion-level guard)
- [ ] Báo cáo trong Section 4.1 và Algorithm 2 của manuscript về protocol mới
- [ ] So sánh accuracy "biased" (cũ) vs "nested" (mới) — minh bạch số liệu bị thổi bao nhiêu
- [ ] Re-run Figure 6 với protocol mới

**Lưu ý cho AI khi implement:** Đảm bảo `random_state` được pass nhất quán, và `StratifiedShuffleSplit` được dùng (không phải `train_test_split` thông thường) để giữ class balance cho multi-class (Digits, Fashion, Wine).

---

### 🔴 T2 — Statistical analysis qua nhiều lần chạy GA độc lập

**Liên kết:** Major comment #3

**Vấn đề:** GA stochastic, paper thừa nhận "instabilities across runs" nhưng không có std/CI/significance test.

**Mục tiêu cụ thể:**
- Chạy ≥ 5 lần GA độc lập với seed khác nhau cho mỗi (dataset, kernel, config)
- Báo cáo `mean ± std` cho mọi accuracy number trong Figure 4, 6, 7
- Thêm significance test khi so sánh GA-QSVM vs baselines (Wilcoxon signed-rank hoặc paired t-test)

**Files cần tạo:**
- `src/ga_qsvm/stats/multirun.py`
- `src/ga_qsvm/stats/significance.py`
- `experiments/exp03_statistical_runs.py`

**Implementation hint:**

```python
# src/ga_qsvm/stats/multirun.py
import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class MultiRunResult:
    accuracies: list   # n_runs values
    mean: float
    std: float
    ci_95_low: float
    ci_95_high: float
    best_circuit_per_run: list

def run_multiple_ga(
    ga_factory: Callable,    # function returning fresh GA instance
    n_runs: int = 5,
    seeds: list = None,
) -> MultiRunResult:
    seeds = seeds or list(range(42, 42 + n_runs))
    accs, circuits = [], []
    for seed in seeds:
        ga = ga_factory(seed=seed)
        best_circ, acc = ga.run()
        accs.append(acc)
        circuits.append(best_circ)
    mean, std = float(np.mean(accs)), float(np.std(accs, ddof=1))
    ci_low, ci_high = _bootstrap_ci(accs, alpha=0.05)
    return MultiRunResult(accs, mean, std, ci_low, ci_high, circuits)
```

```python
# src/ga_qsvm/stats/significance.py
from scipy.stats import wilcoxon, ttest_rel

def compare_methods(scores_a, scores_b, method='wilcoxon'):
    """
    Paired test giữa 2 methods trên cùng seeds.
    Returns p-value và effect size.
    """
    assert len(scores_a) == len(scores_b)
    if method == 'wilcoxon':
        stat, p = wilcoxon(scores_a, scores_b)
    else:
        stat, p = ttest_rel(scores_a, scores_b)
    effect = (np.mean(scores_a) - np.mean(scores_b)) / np.std(scores_b)
    return {'p_value': p, 'effect_size_cohens_d': effect}
```

**Acceptance criteria:**
- [ ] Mọi accuracy number trong paper là `mean ± std` over ≥ 5 runs
- [ ] Figure 4 có shaded region (mean ± std) thay vì single curve
- [ ] Figure 6 có error bars
- [ ] Bảng so sánh GA-QSVM vs RBF/FQK/PQK có cột p-value
- [ ] Section 6 thảo luận về variability quan sát được

**Tip cho AI:** Tổng số runs có thể bùng nổ: 5 datasets × 3 kernels (FQK/PQK/RBF) × 5 qubit configs × 5 seeds = 375 chạy GA. Cần checkpoint mỗi run thành JSON riêng, dùng `joblib.Parallel` hoặc `dask` để phân tán.

---

### 🟡 T3 — Citations & viết lại Section 2

**Liên kết:** Major comment #7

**Mục tiêu cụ thể:** Thêm 3 citations bị thiếu, viết lại trình tự lịch sử cho đúng.

**Citations cần thêm:**

| Tác giả | Bài | DOI / ArXiv | Vị trí trong manuscript |
|---|---|---|---|
| Altares-López, Ribeiro & García-Ripoll (2021) | Automatic design of quantum feature maps | 10.1088/2058-9565/ac1ab1 | Section 2 (đứng trước [20] và [22]) |
| Chen & Chern (2022) | Generating quantum feature maps for SVM classifier | arXiv:2207.11449 | Section 2 (giữa Altares-López và Pellow-Jarman) |
| Moretti et al. (2025) | Enhanced feature encoding and classification on distributed quantum hardware | 10.1088/2632-2153/adb4bc | Section 2 + thảo luận trong Section 6 (scalability) |

**Đoạn template cho Section 2 (chèn vào dòng "In Ref. [12], the authors applied GA..."):**

> The use of GA for QSVM feature map design was pioneered by Altares-López, Ribeiro, and García-Ripoll [NEW1], who proposed an evolutionary scheme to discover compact quantum embeddings outperforming hand-designed ansatze on tabular benchmarks. This line of work was extended by Chen and Chern [NEW2], who proposed an alternative encoding strategy bridging the gap between [NEW1] and later multi-objective approaches. Building on these, Creevey et al. [12] applied GA to search gate-sequence encodings...

**Section 6 — thêm đoạn về Moretti et al.:**

> Recent work by Moretti et al. [NEW3] explicitly addresses hardware execution by parallelizing GA-QSVM across distributed superconducting QPUs, providing a complementary scalability path to our flexible-metadata approach. Integrating their distributed execution model with our adaptive framework is a promising future direction.

**Acceptance criteria:**
- [ ] Bibliography file (`.bib`) cập nhật với 3 entries mới
- [ ] Section 2 có trình tự thời gian đúng: Altares-López → Chen & Chern → Creevey [12] → Pellow-Jarman [20] → Wang [22]
- [ ] Section 6 thảo luận Moretti trong context scalability

---

### 🔴 T4 — Noise-aware simulation

**Liên kết:** Major comment #1 (a) và (b)

**Mục tiêu cụ thể:**
- Chạy lại best GA-optimized circuits dưới noise model thực tế của một backend cụ thể (vd `ibm_kyiv`, `ibm_brisbane`)
- So sánh:
  1. Noiseless simulation (kết quả hiện tại)
  2. Noisy simulation (depolarizing + thermal relaxation + readout error)
  3. Fixed ansatz (ZZ, HEE) dưới cùng noise model
- Đánh giá resilience: bao nhiêu accuracy drop khi thêm noise?

**Files cần tạo:**
- `src/ga_qsvm/kernels/noise_aware.py`
- `experiments/exp01_noise_simulation.py`

**Implementation hint:**

```python
# src/ga_qsvm/kernels/noise_aware.py
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

def get_device_noise_model(backend_name='ibm_brisbane'):
    """Pull realistic noise model từ IBM backend."""
    service = QiskitRuntimeService()           # cần token (có thể dùng free tier)
    backend = service.backend(backend_name)
    noise_model = NoiseModel.from_backend(backend)
    coupling_map = backend.coupling_map
    basis_gates = noise_model.basis_gates
    return noise_model, coupling_map, basis_gates

def noisy_aer_simulator(backend_name='ibm_brisbane', shots=1024):
    noise_model, coupling_map, basis_gates = get_device_noise_model(backend_name)
    sim = AerSimulator(
        noise_model=noise_model,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
    )
    return sim
```

```python
# experiments/exp01_noise_simulation.py
"""
Re-evaluate best GA-evolved circuits dưới noise model.
So sánh với fixed ansatz (ZZFeatureMap, HEE) trên cùng noise model.
"""
def run():
    noisy_sim = noisy_aer_simulator(backend_name='ibm_brisbane', shots=4096)
    for dataset in ['digits', 'wine', 'breast_cancer']:
        for circ_source in ['GA_FQK', 'GA_PQK', 'ZZFeatureMap', 'HEE']:
            acc_noiseless = evaluate(circ_source, dataset, sim=ideal_sim)
            acc_noisy    = evaluate(circ_source, dataset, sim=noisy_sim)
            log_result(dataset, circ_source, acc_noiseless, acc_noisy)
```

**Acceptance criteria:**
- [ ] Thêm 1 figure mới: bar chart so sánh noiseless vs noisy cho mỗi (dataset, method)
- [ ] Thêm bảng: drop tuyệt đối và % drop khi thêm noise
- [ ] Thảo luận: GA-evolved circuits có **resilient hơn** fixed ansatz không?
- [ ] Section 5 có sub-section mới: "5.4 Noise-aware evaluation"
- [ ] Abstract cập nhật để reflect noise analysis

**Lưu ý:** Có thể chạy noise simulation trên CPU bằng `AerSimulator` mà không cần IBM Quantum access — chỉ cần `qiskit-ibm-runtime` để pull `NoiseModel.from_backend()` một lần và cache local.

---

### 🟡 T5 — Computational cost tracking

**Liên kết:** Major comment #2

**Mục tiêu cụ thể:** Báo cáo:
- Wall-clock time cho 1 lần chạy GA full (per dataset, per config)
- Số circuit evaluations
- Scaling theo `n_qubits`, `d`, `n_circuits`, `n_generation`
- So sánh với baseline: classical SVM, fixed-ansatz QSVM

**Files cần tạo:**
- `src/ga_qsvm/utils/timing.py`
- `src/ga_qsvm/utils/circuit_counter.py`
- `experiments/exp06_compute_cost.py`

**Implementation hint:**

```python
# src/ga_qsvm/utils/timing.py
import time, json
from contextlib import contextmanager
from dataclasses import dataclass, asdict

@dataclass
class TimingRecord:
    label: str
    wall_seconds: float
    n_circuit_evals: int
    n_kernel_entries: int
    metadata: dict

@contextmanager
def timed_block(label, counter):
    t0 = time.perf_counter()
    counter.start(label)
    yield counter
    elapsed = time.perf_counter() - t0
    counter.finish(label, elapsed)

class CircuitEvalCounter:
    """Hook vào QSVM training để đếm exact số kernel evals."""
    def __init__(self):
        self._counts = {}
        self._times = {}
    def increment(self, label, k=1):
        self._counts[label] = self._counts.get(label, 0) + k
    def dump(self, path):
        with open(path, 'w') as f:
            json.dump({'counts': self._counts, 'times': self._times}, f, indent=2)
```

**Acceptance criteria:**
- [ ] Thêm bảng vào Section 5: wall-clock per method per dataset (CPU-hours)
- [ ] Thêm figure: scaling plot — `n_qubits` (x-axis) vs `time` (y-axis), log-log
- [ ] Discussion trong Section 6: "GA-QSVM cost X× higher than fixed QSVM but Y× cheaper than random search achieving same accuracy"

---

### 🟡 T6 — Baseline comparison: Random + Greedy circuit search

**Liên kết:** Major comment #5

**Vấn đề:** Phải chứng minh GA hiệu quả hơn các strategies đơn giản, không chỉ hơn ansatz cố định.

**Mục tiêu cụ thể:** Implement và benchmark:
1. **Random circuit sampling** — sample `N` random circuits với cùng budget eval, chọn best
2. **Greedy/incremental construction** — bắt đầu từ circuit rỗng, mỗi step thêm 1 gate maximizing fitness
3. (Optional) **Simulated annealing** / **Local search**

Tất cả với cùng compute budget (số kernel evals) như GA.

**Files cần tạo:**
- `src/ga_qsvm/baselines/random_search.py`
- `src/ga_qsvm/baselines/greedy_search.py`
- `src/ga_qsvm/baselines/fixed_ansatz.py`
- `experiments/exp04_baselines_comparison.py`

**Implementation hint:**

```python
# src/ga_qsvm/baselines/random_search.py
import random
from ga_qsvm.ga.metadata import Metadata
from ga_qsvm.utils.circuit_factory import random_circuit

def random_search(
    metadata: Metadata,
    fitness_fn,
    budget_evals: int,   # phải bằng số evals của GA tương ứng
):
    best, best_fit = None, -1
    for _ in range(budget_evals):
        circ = random_circuit(
            n_qubits=metadata.n,
            depth=metadata.d,
            gate_pool=['Rx','Ry','Rz','H','CX'],
        )
        f = fitness_fn(circ)
        if f > best_fit:
            best, best_fit = circ, f
    return best, best_fit
```

```python
# src/ga_qsvm/baselines/greedy_search.py
def greedy_search(metadata, fitness_fn, max_depth):
    """Empty circuit → add gates greedily, accept if fitness improves."""
    circuit = empty_circuit(metadata.n)
    current_fit = fitness_fn(circuit)
    for depth_step in range(max_depth):
        best_addition, best_fit = None, current_fit
        for gate in ['Rx','Ry','Rz','H','CX']:
            for qubit in range(metadata.n):
                candidate = append_gate(circuit, gate, qubit)
                f = fitness_fn(candidate)
                if f > best_fit:
                    best_addition, best_fit = candidate, f
        if best_addition is None:
            break
        circuit, current_fit = best_addition, best_fit
    return circuit, current_fit
```

**Acceptance criteria:**
- [ ] Bảng so sánh trong Section 5: GA-QSVM vs Random vs Greedy vs Fixed
- [ ] Cùng compute budget — fair comparison
- [ ] Statistical test: GA-QSVM có significantly tốt hơn random/greedy?
- [ ] Nếu KHÔNG significantly tốt hơn → cần discuss thẳng thắn (đây không phải lý do để bị reject; reviewer nói rõ là cần "substantiate the claim", honest negative comparison vẫn ok)

---

### 🟡 T7 — Mở rộng transfer learning + Dataset complexity classification

**Liên kết:** Referee 1 Major #6 + **Referee 2 #1**

**Referee 2 yêu cầu mới:** Phân loại datasets theo complexity (dùng PCA variance decay) **trước khi** transfer; nghiên cứu liệu circuit từ high-complexity datasets có transfer tốt hơn sang similar datasets không; phân tích **circuit characteristics** (depth, CX count) để giải thích transferability.

**Ba lựa chọn (chọn 1):**

#### Option A — Mở rộng phân tích đầy đủ (effort 5–7 ngày, mạnh nhất)

**A.1 Complexity classification của datasets:**

```python
# src/ga_qsvm/transfer/source_selection.py
import numpy as np
from sklearn.decomposition import PCA

def compute_complexity_score(X, target_variance=0.95):
    """
    Complexity = số PCA components cần để retain target_variance.
    Cao = phức tạp, thấp = đơn giản.
    """
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumvar >= target_variance) + 1
    return {
        'n_components_95': n_components,
        'effective_rank': np.sum(pca.explained_variance_ratio_ > 0.01),
        'variance_decay_slope': _fit_decay_slope(pca.explained_variance_ratio_),
    }

# Áp dụng cho 4 datasets từ Figure 3 của paper:
# Wine, Breast Cancer → 10 components (low complexity)
# Digits → 30 components (medium)
# Fashion → 200 components (high)
```

**A.2 Full source × target transfer matrix:**

| Source ↓ / Target → | Digits | Fashion | Wine | Breast Cancer |
|---|---|---|---|---|
| Digits (medium) | original | TL | TL | TL |
| Fashion (high) | TL | original | TL | TL |
| Wine (low) | TL | TL | original | TL |
| Breast Cancer (low) | TL | TL | TL | original |

→ Mỗi cell có FQK và PQK = 32 experiments, × 5 seeds = 160 runs

**A.3 Circuit characteristics analysis cho mỗi transferred circuit:**

```python
# src/ga_qsvm/transfer/circuit_analysis.py
def analyze_circuit(circuit):
    return {
        'depth': circuit.depth(),
        'n_cx': circuit.count_ops().get('cx', 0),
        'n_single_qubit': sum(circuit.count_ops().get(g, 0)
                              for g in ['rx','ry','rz','h']),
        'cx_to_single_ratio': ...,
        'entanglement_pattern': _extract_cx_topology(circuit),
        'expressibility': _meyer_wallach_entanglement(circuit),
    }

# Correlate: complexity(source) vs accuracy(target) vs circuit_metrics
```

**A.4 Test hypothesis của Referee 2:**

> "Circuits evolved from high-complexity datasets transfer better to similar datasets"

→ Plot accuracy improvement (transferred vs random baseline) trên trục `|complexity(source) − complexity(target)|`. Hypothesis: nếu hypothesis đúng, sẽ thấy improvement cao nhất khi complexity gần nhau.

#### Option B — Mid-scope (effort 2–3 ngày)

- Chỉ làm complexity classification + 2 sources (Digits, Wine làm representative cho medium/low)
- Circuit metrics cơ bản (depth, CX count)
- Bỏ qua hypothesis full grid

#### Option C — Giảm tone (effort 0.5 ngày)

- Sửa abstract: bỏ "generalizes effectively"
- Sửa conclusion
- Thêm caveat trong Section 5.3
- Trả lời Referee 2 #1: "We agree this is a valuable direction and have outlined an expanded analysis in future work [...]"

**Đoạn template cho Option C:**

> ~~Furthermore, transfer learning results indicate that GA-QSVM's circuits generalize effectively across datasets.~~
> **→** "As a preliminary investigation, we explore the transferability of GA-evolved circuits across datasets. While circuits trained on Digits show competitive performance when transferred to Wine and Breast Cancer, the magnitude of improvement on Fashion-MNIST is modest (68.1% vs classical 65.7%). A systematic study correlating source–target dataset complexity (e.g., via PCA variance decay) with transferability is required to claim general effectiveness, which we leave for future work."

**Files cần tạo (Option A/B):**
- `src/ga_qsvm/transfer/source_selection.py`
- `src/ga_qsvm/transfer/circuit_analysis.py`
- `experiments/exp05_extended_transfer.py`

**Recommendation phụ thuộc deadline:**
- Có extension đến tháng 8–9: **Option A** (mạnh nhất, address cả 2 reviewers)
- Có extension đến cuối tháng 6/đầu tháng 7: **Option B**
- Không có extension (hard deadline 01/06): **Option C** + commitment trong response letter

---

### 🟢 T8 — Real hardware execution (OPTIONAL — bonus)

**Liên kết:** Major comment #1 (b)

**Mục tiêu:** Chạy best circuit của Digits trên IBM Quantum free tier (`ibm_brisbane` hoặc `ibm_kyiv`).

**Strategy:**
- Chỉ cần 1 dataset (Digits) và 1 kernel (FQK)
- Tính kernel matrix elements `K(x_i, x_j)` cho 100 train + 100 test pairs → 10k kernel entries → batch lên backend
- Dùng `Qiskit Runtime Primitives V2` với `SamplerV2`

**Lưu ý budget:** Free tier IBM cho 10 phút/tháng. Mỗi shot ~ ms. Cần tính toán cẩn thận:
- 10k kernel entries × 1024 shots × ~1ms ≈ 170 phút (vượt free tier)
- **Tactic:** Giảm còn 50 train + 50 test = 2.5k entries × 1024 shots ≈ 40 phút

**Files cần tạo:**
- `experiments/exp02_hardware_ibmq.py`

**Acceptance criteria:**
- [ ] Nhỏ subset chạy thành công trên hardware
- [ ] Accuracy report kèm comparison: ideal vs noisy sim vs hardware
- [ ] Discussion về sai lệch

---

### 🟡 T9 — Backend constraints discussion

**Liên kết:** Major comment #1 (c)

**Mục tiêu:** Thêm vào Discussion thảo luận về cách GA có thể extend để incorporate hardware constraints.

**Nội dung cần thêm vào Section 6:**

1. **Qubit connectivity:** Hiện GA giả định all-to-all connectivity. Thực tế các backend có heavy-hex (IBM) hoặc linear (early IonQ). CX giữa qubits không kề nhau → transpilation chèn SWAP gates → depth tăng.
2. **Native gate set:** GA dùng `{Rx, Ry, Rz, H, CX}` nhưng IBM native là `{RZ, X, √X, ECR}`. Mọi `Ry`, `H` phải decompose → thêm gates.
3. **Transpilation overhead:** Cần đo `circuit_depth_after_transpile / circuit_depth_logical` ratio.
4. **Extension hướng tới hardware-aware GA:**
   - Thêm fitness term: `accuracy − λ * transpiled_depth`
   - Hoặc restrict gate pool theo native set của backend cụ thể
   - Hoặc dùng coupling map làm constraint khi sample CX

**Acceptance criteria:**
- [ ] Sub-section mới trong Section 6: "6.2 Hardware-aware extensions"
- [ ] Code stub (không cần chạy) cho hardware-aware fitness:

```python
def hardware_aware_fitness(circuit, kernel_fn, backend, lambda_=0.01):
    accuracy = kernel_fn(circuit)
    transpiled = transpile(circuit, backend=backend, optimization_level=3)
    return accuracy - lambda_ * transpiled.depth()
```

---

### 🟢 T10 — Hyperparameter summary table

**Liên kết:** Minor comment #4

**Mục tiêu:** Bảng tóm tắt giá trị Metadata `M` cho TỪNG experiment.

**Template:**

| Experiment | Figure | n | n_Rx | n_Ry | n_Rz | d | n_circuits | n_gen | p_m | τ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hyperparam sweep — depth | 4(a) | 5 | – | – | – | {5,10,15,20,25} | 16 | 200 | 0.1 | 0.8 | nCX=2n |
| Hyperparam sweep — population | 4(b) | 5 | – | – | – | 5 | {4,8,16,20} | 200 | 0.1 | 0.8 | nCX=2n |
| Hyperparam sweep — nCX | 4(c) | 5 | – | – | – | 5 | 16 | 200 | 0.1 | 0.8 | nCX∈{5,10,15,20,25} |
| Hyperparam sweep — mutation | 4(d) | 5 | – | – | – | 5 | 16 | 200 | {0.001,0.01,0.1,0.3,0.5} | 0.8 | nCX=2n |
| Main results | 6 | 3–7 | * | * | * | 5 | 16 | full | 0.1 | 0.8 | nCX=2n |
| Transfer | 7 | 10 | * | * | * | TBD | 16 | full | 0.1 | 0.8 | source=Digits |

(* = decided by GA optimization)

**Acceptance criteria:**
- [ ] Bảng được đặt ngay đầu Section 5 hoặc trong Appendix
- [ ] Mọi experiment trong paper trace được về 1 row của bảng

---

### 🟢 T11 — Discuss negative results

**Liên kết:** Minor comment #5

**Mục tiêu:** Figures 4(c) và 4(d) cho thấy `n_CX` và `p_m` không ảnh hưởng nhiều đến fitness — đây là *interesting negative result*.

**Đoạn cần thêm vào Section 5.2:**

> "Figures 4(c)–(d) reveal a notable insensitivity of final fitness to both the CX-gate budget `n_CX` and mutation probability `p_m` within the tested ranges. We interpret this as follows. (i) For `n_CX`: once enough entanglement is available beyond a saturation threshold (~`2n`), additional CX gates do not improve expressivity for these tabular datasets. (ii) For `p_m`: the elitist selection strategy dominates the search dynamics, so mutation primarily contributes diversity rather than direct fitness improvements. **Practical implication:** A simpler single-objective GA without exhaustive tuning of `n_CX` and `p_m` would likely suffice; future hyperparameter optimization effort should focus on `d` (circuit depth) and `n_circuits` (population size), which show clear monotonic effects in Figures 4(a)–(b)."

---

### 🟢 T12 — Moderate "generalizes effectively" claim

**Liên kết:** Minor comment #6

Đã include trong T7 Option B. Cần áp dụng nhất quán ở:
- [ ] Abstract
- [ ] Section 5.3 final paragraph
- [ ] Section 6 conclusion

---

### 🟢 T13 — Giải thích thật lý do loại Fashion-MNIST khỏi GA-QSVM training

**Liên kết:** Minor comment #2

**Vấn đề:** Paper nói "huge number of instances" — không thuyết phục khi chỉ dùng 100 samples.

**Đoạn cần thay trong Section 5.1:**

> ~~Only three datasets, including Digits, Wine, and Breast Cancer were used in GA-QSVM because of the huge number of Instances (#Instances) from Fashion.~~
> **→** "Fashion-MNIST was excluded from the GA-QSVM training experiments not because of dataset size — we use only 100 samples for all datasets — but because of qubit-count requirements. Retaining 95% explained variance on Fashion-MNIST requires 200 PCA components (Figure 3b), which would necessitate a 200-qubit circuit under angle encoding. This is far beyond what is tractable on current simulators or NISQ devices. We instead use Fashion-MNIST as a transfer-learning target with 10 features (Section 5.3) to test whether circuits learned on lower-dimensional sources can extrapolate."

---

### 🟢 T14 — Discuss sample size implications

**Liên kết:** Minor comment #1

**Mục tiêu:** Thừa nhận limitation, thảo luận hệ quả.

**Đoạn cần thêm vào Section 6:**

> "Our experiments use only 100 training and 100 testing samples per dataset due to the O(N²) scaling of kernel-based methods under quantum simulation. This regime may artificially favor quantum kernels over classical ones, since: (i) RBF kernels typically benefit from larger N to estimate optimal bandwidth, and (ii) the small-N regime emphasizes feature-map quality over data quantity. A systematic study of GA-QSVM accuracy as N grows from 100 to 1000+ is left as future work, contingent on hardware acceleration or distributed evaluation as in Moretti et al. [NEW3]."

---

### 🟢 T15 — Clarify theoretical complexity claim

**Liên kết:** Minor comment #3

**Đoạn cần thêm sau Equation/claim về O(M^4.67/ε²) trong Section 1:**

> "We note that this theoretical complexity advantage is asymptotic and assumes quantum random access to data. In the simulated regime of our experiments (M=100, n=7 qubits, exact statevector simulation), the quoted speed-up is not realized; the classical simulator computes kernels in O(M² · 2^n) which is, in fact, less favorable than classical SVM for these parameters. The theoretical bound becomes relevant only on actual quantum hardware at significantly larger M, motivating future hardware-scale studies."

---

### 🟢 T16 — Proofread + fix typos

**Liên kết:** Minor comment #7

**Lỗi cụ thể được flag bởi reviewer:**
- [ ] Section 5.1: "a classical computer.." → "a classical computer."
- [ ] Inconsistent spacing in equations
- [ ] Fix all "Figure. X" → "Figure X" (chuẩn IOP)
- [ ] Spelling: "Pello et al." trong Section 2 không match với "Pellow-Jarman et al." trong references — nhất quán hóa
- [ ] Check capitalization: "QSVM" vs "qsvm" trong các đoạn
- [ ] Math notation: "x⊺i xj" — đảm bảo transpose syntax nhất quán
- [ ] Period at end of every equation/sentence

**Tool đề xuất:**
- LaTeX: `chktex`, `lacheck`
- Grammar: Grammarly, LanguageTool
- Sao chép sang Word → review tracking on

---

### 🟡 T17 — Qiskit standard fixed-ansatz baselines

**Liên kết:** Referee 2 comment #2

**Vấn đề:** Hiện chỉ so sánh với `ZZFeatureMap` và `HEE`. Referee 2 yêu cầu thêm các feature maps có sẵn của Qiskit để chứng minh GA-QSVM thực sự outperform các fixed embeddings phổ biến.

**Mục tiêu cụ thể:** Thêm baselines:
1. **`PauliFeatureMap`** — generalization của ZZ, với configurable Pauli strings
2. **`EfficientSU2`** — hardware-efficient ansatz từ `qiskit.circuit.library`
3. **`TwoLocal`** — generic 2-local ansatz với configurable rotation + entanglement blocks

Tất cả chạy trên cùng datasets (Digits, Wine, Breast Cancer) với cùng `n_qubits`.

**Files cần tạo:**
- `src/ga_qsvm/baselines/qiskit_ansatzes.py`
- Cập nhật `experiments/exp04_baselines_comparison.py` để include thêm 3 ansatz này

**Implementation hint:**

```python
# src/ga_qsvm/baselines/qiskit_ansatzes.py
from qiskit.circuit.library import (
    PauliFeatureMap, EfficientSU2, TwoLocal,
    ZZFeatureMap, ZFeatureMap
)

def get_baseline_feature_maps(n_qubits, reps=2):
    """
    Returns dict of standard Qiskit feature maps cho fair comparison.
    Mọi map đều có depth tương đương để comparison được fair.
    """
    return {
        'ZFeatureMap':    ZFeatureMap(feature_dimension=n_qubits, reps=reps),
        'ZZFeatureMap':   ZZFeatureMap(feature_dimension=n_qubits, reps=reps,
                                       entanglement='linear'),
        'PauliFeatureMap': PauliFeatureMap(feature_dimension=n_qubits, reps=reps,
                                           paulis=['Z', 'XX', 'YY']),
        'EfficientSU2':   EfficientSU2(num_qubits=n_qubits, reps=reps,
                                       entanglement='linear'),
        'TwoLocal':       TwoLocal(num_qubits=n_qubits, reps=reps,
                                   rotation_blocks=['ry', 'rz'],
                                   entanglement_blocks='cx',
                                   entanglement='linear'),
    }

def benchmark_all_ansatzes(X_train, y_train, X_test, y_test, n_qubits):
    """Run QSVM với mỗi fixed ansatz, return dict of accuracies."""
    results = {}
    for name, fmap in get_baseline_feature_maps(n_qubits).items():
        for kernel_type in ['FQK', 'PQK']:
            acc = evaluate_qsvm(fmap, kernel_type, X_train, y_train, X_test, y_test)
            results[f'{name}_{kernel_type}'] = acc
    return results
```

**Lưu ý fair comparison:**
- **Depth tương đương:** Cùng số `reps` để mọi ansatz có circuit depth tương đương với GA output
- **Cùng số parameters:** Note rằng `EfficientSU2` và `TwoLocal` là *variational* (có trainable params), khác với *feature map* (không train). Khi dùng làm feature map, fix params hoặc dùng angle-encoding cho data
- **Cùng kernel type:** Test cả FQK và PQK

**Acceptance criteria:**
- [ ] Bảng mở rộng trong Section 5.3 (hoặc Figure 6 mở rộng): GA-FQK, GA-PQK, ZFM, ZZFM, **PauliFM, EfficientSU2, TwoLocal**, RBF
- [ ] Mean ± std qua ≥ 5 runs cho mỗi method
- [ ] Significance test: GA-QSVM significantly tốt hơn cả 3 ansatz mới?
- [ ] Discussion: ansatz nào competitive nhất? Có pattern không?

---

### 🔴 T18 — PCA leakage audit (CRITICAL)

**Liên kết:** Referee 2 comment #4

**Vấn đề:** Reviewer 2 hỏi cụ thể "specify whether PCA was fitted only on the training data before being applied to the test data". Đây là **methodological red flag** — nếu code hiện tại `pca.fit_transform(X_full)` rồi mới split → **test data leak vào PCA fitting** → mọi accuracy số đều bị bias upward.

Đây là leakage thứ hai (cùng category với T1 — test-set leakage trong GA).

**Mục tiêu cụ thể:**
1. **Audit** code hiện tại trong repo: tìm mọi chỗ gọi `PCA`, `StandardScaler`, normalizer
2. **Sửa** nếu phát hiện leak: PCA chỉ `.fit()` trên `X_train`, sau đó `.transform()` cả train và test
3. **Document** trong manuscript: thêm câu rõ ràng trong Section 5.1
4. **Tương tự cho mọi preprocessing:** scaling, normalization

**Files cần audit:**
- Search trong repo: `grep -rn "PCA\|fit_transform\|StandardScaler" src/ experiments/`
- Đặc biệt check file load datasets

**Pattern ĐÚNG:**

```python
# ĐÚNG — fit chỉ trên train
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Split TRƯỚC
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

# 2. Scaler — fit chỉ trên train
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# 3. PCA — fit chỉ trên train
pca = PCA(n_components=n_components).fit(X_train_s)
X_train_pca = pca.transform(X_train_s)
X_test_pca  = pca.transform(X_test_s)
```

**Pattern SAI (cần fix nếu thấy):**

```python
# SAI — leakage
X_pca = PCA(n_components=k).fit_transform(X)   # fit trên cả full dataset
X_train, X_test = train_test_split(X_pca, ...)  # rồi mới split
```

**Acceptance criteria:**
- [ ] Hoàn thành audit, document trong `docs/pca-audit.md`
- [ ] Nếu phát hiện leak → **re-run mọi experiment** và compare số mới vs cũ
- [ ] Section 5.1 thêm 1 câu rõ: "PCA was fitted exclusively on the training partition and the resulting transformation applied to the test partition; the same protocol was used for any standardization step."
- [ ] Pipeline tốt nhất dùng `sklearn.pipeline.Pipeline` để impossible-by-design leak:

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_components)),
    ('qsvm', QSVMClassifier(circuit=ga_circuit)),
])
pipe.fit(X_train, y_train)   # fit toàn pipeline only trên train
score = pipe.score(X_test, y_test)
```

**Lưu ý:** T1 (GA test-set leakage) và T18 (PCA leakage) là **hai loại leakage độc lập** — cần fix cả hai. Nested CV của T1 không tự động giải quyết T18.

---

### 🟡 T19 — Reproducibility documentation

**Liên kết:** Referee 2 comment #4 (phần còn lại)

**Mục tiêu cụ thể:** Document trong manuscript và repo:
1. **Random seed settings** — seed nào dùng cho gì
2. **Train/test split procedure** — chính xác từng bước
3. **GA initialization process** — population đầu tiên được tạo ra sao
4. **Hyperparameter justification** — vì sao chọn `d=5`, `p_m=0.1`, `n_circuits=16`, v.v.

**Files cần tạo:**
- `docs/reproducibility.md` (chính)
- Cập nhật README.md với link

**Template nội dung `docs/reproducibility.md`:**

```markdown
# Reproducibility Guide

## 1. Random Seeds

Mọi experiment dùng seeds = [42, 43, 44, 45, 46] cho 5 independent runs.

| Component | Seed param | Value |
|---|---|---|
| Train/test split | `random_state` của `StratifiedShuffleSplit` | per-run seed |
| GA initialization | `numpy.random.seed()` | per-run seed |
| QSVM training | sklearn `SVC(random_state=)` | per-run seed |
| Qiskit simulator | `AerSimulator(seed_simulator=)` | per-run seed |

## 2. Data Split Protocol

1. Load dataset → `X_raw, y`
2. Stratified split 60% train, 20% val, 20% test với `random_state=seed`
3. Fit `StandardScaler` trên train → transform train, val, test
4. Fit `PCA(n_components=n)` trên train scaled → transform train, val, test
5. GA fitness eval: train QSVM trên train_pca, score trên val_pca
6. Final eval: train trên (train+val)_pca, score 1 lần trên test_pca

## 3. GA Initialization

- Population đầu tiên: `n_circuits` random circuits
- Mỗi circuit: random sample `d` layers từ gate pool {Rx, Ry, Rz, H, CX}
- Mỗi gate gán random qubit (CX random 2 qubits)
- Rotation angles: random uniform trong [0, 2π] (sẽ thay bằng data encoding)
- Normalizer apply ngay từ generation 0

## 4. Hyperparameter Justification

| Param | Default | Lý do |
|---|---|---|
| `d=5` | depth | Compromise: deeper → better fitness (Fig 4a) nhưng cost cao + noise nặng trên hardware |
| `p_m=0.1` | mutation rate | Saturation: 4(d) cho thấy 0.01–0.5 không khác nhau, chọn 0.1 trung tâm |
| `n_circuits=16` | population | 4(b) cho thấy 16–20 converge ổn, 4 và 8 không đủ diversity |
| `n_CX=2n` | CX budget | 4(c) saturation: thêm CX không giúp; 2n là minimum đủ cho entanglement |
| `n_generation=200` | generations | Fitness saturate ~50–100 gens (Fig 4); 200 để safety margin |
| `τ=0.8` | threshold | Empirically đủ cao để quality, không quá cao gây stuck |
```

**Acceptance criteria:**
- [ ] `docs/reproducibility.md` đầy đủ trên repo
- [ ] Section 5.1 và 5.2 manuscript ref đến file này
- [ ] Mọi script có `--seed` argument
- [ ] CI test reproducibility: chạy với seed cố định 2 lần → kết quả giống nhau bit-by-bit

---

## 4. Roadmap — Hai phương án

### 4-bis. Crash plan 13 ngày (nếu KHÔNG xin extension)

**Mục tiêu:** Survival mode — chỉ làm những gì tuyệt đối cần thiết để không bị reject thẳng.

```
Day 1-2:   T18 (PCA leakage audit) — CRITICAL, làm trước nhất
           T1 (test-set leakage fix với nested CV)
Day 3-4:   T2 (multi-run stats) — chạy parallel trên tất cả CPU cores
           T3 (citations) — viết trong khi T2 chạy
Day 5-6:   T4 (noise simulation) — sub-set datasets thôi, không full grid
Day 7-8:   T17 (Qiskit fixed-ansatz baselines) — chỉ FQK, skip PQK nếu kẹt
Day 9-10:  T5 (compute cost) + T11, T13, T15 (minor textual fixes)
           T7 Option C (giảm tone transfer learning)
Day 11:    T9, T10, T12, T14 (manuscript editing)
           T19 (reproducibility doc)
Day 12:    T16 (proofread) + regenerate figures
           Viết point-by-point response
Day 13:    Final review, upload all files, submit
```

**Drop hoàn toàn:** T6 (random/greedy baselines), T7 Option A/B (extended TL), T8 (hardware).
Trong response letter, commit các items dropped vào "future work" và giải thích tại sao không làm được trong 13 ngày.

⚠️ **Rủi ro:** Reviewer có thể không hài lòng và đề xuất reject vì các major comments không được address đầy đủ.

### 4-tris. Full plan 6 tuần (nếu CÓ extension)

```
Week 1:  T18 (PCA audit) + T1 (test leakage) + T3 (citations)
         + T10, T13 (quick wins)
Week 2:  T2 (multi-run stats) — chạy parallel
         + T19 (reproducibility doc)
Week 3:  T4 (noise simulation) + T9 (backend discussion)
Week 4:  T17 (Qiskit baselines) + T6 (random/greedy baselines)
Week 5:  T5 (compute cost) + T7 Option A hoặc B (transfer learning extended)
Week 6:  T8 (optional hardware) + T11–T16 + regenerate figures
         + draft revision response + final polish
```

**Recommendation mạnh:** Xin extension đến **31/07/2026** hoặc **31/08/2026** để có buffer an toàn.

---

## 5. Checklist deliverables cuối cùng (theo Revision-checklist của IOP)

### Required files

- [ ] **Author Response** (`docs/revision-response.md` → export PDF)
  - [ ] Point-by-point trả lời từng comment (7 major + 7 minor = 14 points)
  - [ ] Anonymised (nếu double-blind)
  - [ ] Upload ở Step 1
- [ ] **Highlighted PDF** (revised manuscript with changes highlighted)
  - [ ] Designation: "Complete Document for Review (PDF Only)"
  - [ ] Anonymised (nếu double-blind)
  - [ ] Figures + tables included
- [ ] **Source File** (revised TeX/Word, clean)
  - [ ] Full author list + affiliations
  - [ ] Corresponding author + email
  - [ ] Funding/acknowledgements
  - [ ] Ethics statement (nếu cần)
  - [ ] Tables/figures/equations editable
- [ ] **Clean PDF version** (unmarked, từ clean source)

### Optional/recommended

- [ ] **Supplementary material:** Extra figures (noise comparison, baseline comparison, scaling plots) + bảng full hyperparameter sweep mean/std
- [ ] **Additional source files:** High-res figures cho production

### Repo deliverables (parallel work)

- [ ] README cập nhật: hardware notes, citation, reproducibility instructions
- [ ] `docs/reproducibility.md`: exact commands để regenerate mỗi figure
- [ ] Tag release `v2.0-revised` trên GitHub
- [ ] Zenodo DOI cho version submission

---

## 6. Notes cuối cho AI

Khi dùng tài liệu này làm context:

1. **Ưu tiên T1, T2, T3, T4** — đây là 4 nhiệm vụ critical, không làm thì gần như chắc chắn bị reject.
2. **Mọi code mới phải có:** type hints, docstrings, `random_state` parameter, có thể chạy headless (no plt.show()).
3. **Cấu trúc output mỗi experiment:** `results/raw/<exp_name>/<dataset>/<config_hash>/<seed>.json` — không overwrite, cho phép aggregate sau.
4. **Tránh:** mix logic GA với logic plotting trong cùng file. Tách thành (run experiment → save JSON) + (load JSON → render figure).
5. **Test parity:** Trước mỗi major refactor, save 1 reference run từ codebase cũ; sau refactor verify accuracy match (within stochastic noise).

---

*File này được tạo dựa trên: review của Referee 1 (MLST-104723_review.pdf), review của Referee 2 (trong decision letter ngày 19/05/2026), revision checklist của IOP, và manuscript hiện tại (arXiv:2511.19160v1).*

---

## 7. Template email xin extension

**Gửi tới:** mlst@ioppublishing.org
**CC:** Jaya Srivastava (editorial assistant)
**Subject:** MLST-104723 — Request for revision deadline extension

```
Dear Dr. Srivastava,

Re: Manuscript MLST-104723 — "Flexible Genetic Algorithm for Quantum Support
Vector Machines"

Thank you very much for sending the reviewer reports. We are grateful to both
referees for their detailed and constructive feedback, which we believe will
substantially strengthen the manuscript.

After careful study of both reports, we have identified the following work
required for a thorough revision:

  (i)   Re-running all experiments with multi-seed statistical analysis
        (Referee 1, Major #3 and Referee 2, point 4) — approximately 375
        independent GA runs across datasets, configurations, and seeds;
  (ii)  Implementing noise-aware simulations using realistic backend noise
        models (Referee 1, Major #1);
  (iii) Auditing and correcting the PCA fitting protocol to ensure no data
        leakage (Referee 2, point 4);
  (iv)  Re-architecting the GA fitness evaluation to eliminate test-set
        leakage via nested cross-validation (Referee 1, Major #4);
  (v)   Adding baseline comparisons against random circuit search, greedy
        construction, and additional Qiskit feature maps (PauliFeatureMap,
        EfficientSU2, TwoLocal) (Referee 1 Major #5; Referee 2 point 2);
  (vi)  Expanding the transfer learning analysis with dataset-complexity
        classification (Referee 2, point 1).

These items collectively require substantial new experimentation that cannot
be completed by the current deadline of 01 June 2026 while maintaining the
scientific rigor that the referees rightly request.

We therefore respectfully request an extension of the revision deadline to
**[DD Month 2026]**. This will allow us to address all reviewer comments
fully rather than partially.

We are committed to submitting a thoroughly revised manuscript and appreciate
your consideration.

Best regards,
[Corresponding author name]
On behalf of the authors
```

**Đề xuất ngày deadline mới:**
- Conservative: **31/07/2026** (~10 tuần) — phù hợp với full 6-tuần plan + buffer
- Comfortable: **31/08/2026** (~14 tuần) — bao gồm cả T8 (hardware) và phân tích Option A đầy đủ
- Aggressive: **30/06/2026** (~6 tuần) — chỉ đủ cho full plan, không buffer

MLST/IOP thường approve extension đến 3 tháng cho major revision khi lý do hợp lý. Quan trọng là **gửi sớm** (trong vài ngày tới), không đợi đến gần deadline.

---

## 8. Quick reference — Mapping comments ↔ tasks

### Referee 1

| Comment | Task(s) |
|---|---|
| Major #1 (hardware, noise, backend) | T4, T8, T9 |
| Major #2 (computational cost) | T5 |
| Major #3 (statistics, multi-run) | T2 |
| Major #4 (test-set leakage) | T1 |
| Major #5 (baseline comparison) | T6 |
| Major #6 (transfer learning) | T7 |
| Major #7 (citations) | T3 |
| Minor #1 (sample size) | T14 |
| Minor #2 (Fashion exclusion) | T13 |
| Minor #3 (complexity claim) | T15 |
| Minor #4 (hyperparameter table) | T10 |
| Minor #5 (negative results) | T11 |
| Minor #6 ("generalizes effectively") | T12 |
| Minor #7 (proofread) | T16 |

### Referee 2

| Comment | Task(s) |
|---|---|
| #1 (transfer learning + complexity) | T7 (mở rộng) |
| #2 (Qiskit fixed ansatzes) | **T17** |
| #3 (computational cost) | T5 |
| #4 (reproducibility: seeds, splits, **PCA fitting**, hyperparams) | **T18** (PCA leakage) + **T19** (reproducibility doc) + T10 (hyperparams) + T2 (seeds) |
