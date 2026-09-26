# REFERENCES — Academic Literature & Theoretical Foundations

This document serves as the authoritative bibliography of all research papers, PhD/diploma theses, journal articles, and theoretical literature implemented in this codebase.

---

## Table of Contents

1. [Homogeneous Self-Dual Algorithm (HSD)](#1-homogeneous-self-dual-algorithm-hsd)
2. [First-Order Primal-Dual Hybrid Gradient (PDLP)](#2-first-order-primal-dual-hybrid-gradient-pdlp)
3. [MIP Presolve Reductions [AGH]](#3-mip-presolve-reductions-agh)
4. [Constraint Integer Programming & MIP Architecture [CIP]](#4-constraint-integer-programming--mip-architecture-cip)
5. [Dual Simplex Method & Dual Steepest Edge (DSE)](#5-dual-simplex-method--dual-steepest-edge-dse)
6. [Irreducible Infeasible Subsystems (IIS)](#6-irreducible-infeasible-subsystems-iis)
7. [Primal Heuristics for MIP](#7-primal-heuristics-for-mip)
8. [Cutting Plane Separators & Dual Bounds](#8-cutting-plane-separators--dual-bounds)
9. [Interior-Point Methods & Predictor-Corrector Loops](#9-interior-point-methods--predictor-corrector-loops)
10. [Matrix Scaling & Equilibration](#10-matrix-scaling--equilibration)
11. [Basis LU Factorization & Basis Updates](#11-basis-lu-factorization--basis-updates)
12. [General Presolve & Ratio Tests](#12-general-presolve--ratio-tests)
13. [Conflict Analysis & Reliability Branching](#13-conflict-analysis--reliability-branching)

---

## 1. Homogeneous Self-Dual Algorithm (HSD)

* **Primary Citation**: Andersen, E. D., & Andersen, K. D. (2000). *"The MOSEK interior point optimizer for linear programming: an implementation of the homogeneous algorithm"*. In *High Performance Optimization* (pp. 197–232). Springer / Kluwer Academic Publishers. (Cited as **[AA]**)
* **Code References**:
  * [`FORMULATION.md`](FORMULATION.md#L677-L1023)
  * [`../include/sovsolve/solver/HomogeneousNewton.hpp`](../include/sovsolve/solver/HomogeneousNewton.hpp)
  * [`../include/sovsolve/solver/HomogeneousSolve.hpp`](../include/sovsolve/solver/HomogeneousSolve.hpp)
  * [`../include/sovsolve/solver/HomogeneousStep.hpp`](../include/sovsolve/solver/HomogeneousStep.hpp)
  * [`../include/sovsolve/model/Options.hpp`](../include/sovsolve/model/Options.hpp#L547-L555)
* **Specific Sections & Equations Used**:
  * **§1.2 & §13.3a**: Transposition of general finite variable bounds onto the standard-form Homogeneous Linear Formulation (HLF).
  * **§1.4 & Table 1.1**: Step size selection, centering parameter, starting point, and convergence criteria.
  * **§1.4.1**: Predictor step direction ($\gamma = 0, \eta = 1$).
  * **§1.4.2 & Eq. (1.12)**: Centering parameter formula $\gamma = (1-\alpha)^2 \min(1-\alpha, \beta_1)$ with $\beta_1 = 0.1$.
  * **§1.4.3**: Step length reduction $\alpha = \min(\beta_3 \alpha_{\max}, 1)$ with $\beta_2 = 10^{-8}, \beta_3 = 0.9999$.
  * **§1.4.5 & Eq. (1.24)**: Termination tolerances ($\rho_P, \rho_D = 10^{-8}, \rho_A = 10^{-10}, \rho_I, \rho_\mu = 10^{-10}$).
  * **§1.5 & Eq. (1.28)–(1.29)**: Bordered Newton solve using a single $KktSolver$ factorization per iteration via a scalar Schur complement.
  * **Theorems 2 & 3**: Strict complementarity conditions for optimality ($\tau^* > 0$) vs. primal/dual infeasibility certificates ($\tau^* = 0, \kappa^* > 0$).
  * **Equations**: (1.2), (1.12), (1.13), (1.20), (1.21), (1.22), (1.23), (1.24), (1.25)–(1.29).

---

## 2. First-Order Primal-Dual Hybrid Gradient (PDLP)

* **Paper 1 (Main Algorithm)**: Applegate, D., Diaz, M., Hinder, O., Lu, H., Lubin, M., O'Donoghue, B., & Schudy, W. (2021). *"Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient"*. *Advances in Neural Information Processing Systems (NeurIPS 2021)*, 34, 20243–20257.
  * **Code References**: [`../include/sovsolve/solver/pdlp/Pdlp.hpp`](../include/sovsolve/solver/pdlp/Pdlp.hpp), [`HIGHS-COMPARISON.md`](HIGHS-COMPARISON.md).
  * **Sections Used**: Baseline PDHG saddle-point formulation (Eq. 1–3), step size reparameterization $\tau \sigma \|K\|^2 \le 1$ (Eq. 4), termination criteria (Eq. 6a–6c), adaptive step size (Algorithm 2), adaptive restarts (§3.2), and primal weight updates (Algorithm 3).
* **Paper 2 (Restarts & Trust Region)**: Applegate et al. (2021). *"Faster First-Order Primal-Dual Methods for Linear Programming using Restarts and Sharpness"*. arXiv:2105.12715.
  * **Code References**: [`../include/sovsolve/solver/pdlp/DualityGap.hpp`](../include/sovsolve/solver/pdlp/DualityGap.hpp), [`../include/sovsolve/solver/pdlp/TrustRegion.hpp`](../include/sovsolve/solver/pdlp/TrustRegion.hpp).
  * **Sections Used**: §3 & §6.3 (Linear-time normalized duality gap $\rho_r(z)$ evaluation), §6.3.1 & Appendix F (Trust region subproblem over boxed variables, Eq. 49–54, 1D breakpoint search, $O(N)$ active-set halving algorithm).
* **Paper 3 (Infeasibility Certificates)**: Applegate et al. (2021). *"Infeasibility detection with primal-dual hybrid gradient for large-scale linear programming"*. arXiv:2102.04592.
  * **Code References**: [`../include/sovsolve/solver/pdlp/Infeasibility.hpp`](../include/sovsolve/solver/pdlp/Infeasibility.hpp), [`../src/solver/pdlp/Infeasibility.cpp`](../src/solver/pdlp/Infeasibility.cpp).
  * **Sections Used**: §1.1 (Three candidate sequences converging to infimal displacement vector), §4.1 & Lemma 3 & Proposition 4 (Infimal displacement vector $v=(v_x, v_y)$), §6 Equations (50) and (51) ($\epsilon$-approximate certificates measuring constraint violation per unit of objective improvement).

---

## 3. MIP Presolve Reductions [AGH]

* **Citation**: Achterberg, T., Bixby, R. E., Gu, Z., Rothberg, E., & Weninger, D. (2020). *"Presolve reductions in mixed integer programming"*. *Mathematical Programming Computation*, 12(4), 549–603. (Cited as **[AGH]**)
* **Code References**:
  * [`../include/sovsolve/solver/MilpCanonicalPresolve.hpp`](../include/sovsolve/solver/MilpCanonicalPresolve.hpp)
  * [`../src/solver/MilpCanonicalPresolve.cpp`](../src/solver/MilpCanonicalPresolve.cpp)
  * [`../include/sovsolve/model/Options.hpp`](../include/sovsolve/model/Options.hpp#L407)
* **Specific Sections Used**:
  * **§4.5**: Implied Free Variable Substitution (with Markowitz threshold $|a_{ij}| \ge 0.01 \max |a_{\cdot j}|$ and fill-in limits).
  * **§5 (5.3)**: Multi-row reductions & Non-zero cancellation.
  * **§6.1 & §6.3**: Parallel Columns & Minkowski-sum bound aggregation (Eq. 6.1).
  * **§6.4**: Dominated columns.
  * **§7.2 & §7.6**: Probing and Implied integer detection.
  * **§8 & Table 25**: Node presolve metrics.

---

## 4. Constraint Integer Programming & MIP Architecture [CIP]

* **PhD Thesis**: Achterberg, T. (2007). *"Constraint Integer Programming"*. PhD Thesis, Technische Universität Berlin. (Cited as **[CIP]**)
* **Code References**:
  * [`../include/sovsolve/solver/MilpSolve.hpp`](../include/sovsolve/solver/MilpSolve.hpp)
  * [`../include/sovsolve/solver/MilpSeparators.hpp`](../include/sovsolve/solver/MilpSeparators.hpp)
  * [`../include/sovsolve/model/Options.hpp`](../include/sovsolve/model/Options.hpp#L306-L434)
* **Specific Chapters Used**:
  * **Chapter 5 (§5.2, §5.3, §5.4, §5.7, §5.11, Alg. 5.2)**: Branching variable scoring ($q^- \cdot q^+$ product score, pseudocosts, reliability branching).
  * **Chapter 6 (§6.2, §6.3, §6.6)**: Node selection strategies (best bound, best estimate, depth-first plunging).
  * **Chapter 7 (§7.1, §7.3, §7.4)**: Domain propagation, linear constraint propagation, watched-literal scheme.
  * **Chapter 8 (§8.3, §8.4–8.5, §8.8, §8.10, Alg. 3.2)**: Cutting plane separation, Gomory mixed integer (GMI) cuts, local reduced cost strengthening, cut-and-branch loop.
  * **Chapter 9 (§9.1.1, §9.1.2, §9.2, §9.3.3)**: Primal heuristics (simple rounding, shifted rounding, RENS, Feasibility Pump).
  * **Chapter 10 (§10.1, §10.6)**: MIP Presolve stage A.
  * **Chapter 11 (§11.3)**: Conflict analysis & clause aging.

---

## 5. Dual Simplex Method & Dual Steepest Edge (DSE)

* **Dual Simplex Algorithm**: Lemke, C. E. (1954). *"The dual method of solving the linear programming problem"*. *Naval Research Logistics Quarterly*, 1(1), 36–47.
  * **Code References**: [`../include/sovsolve/solver/simplex/DualSimplex.hpp`](../include/sovsolve/solver/simplex/DualSimplex.hpp).
* **Dual Steepest Edge & Perturbation Thesis**: Koberstein, A. (2005). *"The dual simplex method, techniques for a fast and stable implementation"*. PhD Thesis, University of Paderborn.
  * **Code References**:
    * [`../src/solver/simplex/DualSimplex.cpp`](../src/solver/simplex/DualSimplex.cpp)
    * [`../src/solver/simplex/SolveSimplex.cpp`](../src/solver/simplex/SolveSimplex.cpp)
    * [`../include/sovsolve/solver/simplex/Basis.hpp`](../include/sovsolve/solver/simplex/Basis.hpp)
  * **Sections Used**: §3.3 (Dual Steepest Edge weights $\beta_r = \|B^{-T} e_r\|^2$), §6.3.1 (Cost perturbation against dual degeneracy), §8.2.2.1 (DSE weight updates in $w$-space and weight floor), §9.4 (Empirical DSE evaluation).
* **DSE Recurrence Formula**: Forrest, J. J. H., & Goldfarb, D. (1992). *"Steepest-edge simplex algorithms for linear programming"*. *Mathematical Programming*, 57(1-3), 341–374.
  * **Code References**: [`../src/solver/simplex/DualSimplex.cpp`](../src/solver/simplex/DualSimplex.cpp#L245), [`../include/sovsolve/model/Options.hpp`](../include/sovsolve/model/Options.hpp#L764).

---

## 6. Irreducible Infeasible Subsystems (IIS)

* **Paper 1 (IIS Characterization)**: Gleeson, A. C., & Ryan, J. (1990). *"Identifying irreducible infeasible subsystems"*. *ORSA Journal on Computing*, 2(1), 61–63.
* **Paper 2 (SDP & LP Support Vertices)**: Kellner, A., Pfetsch, M. E., & Theobald, T. (2018). *"Irreducible Infeasible Subsystems of Semidefinite Systems"*. arXiv:1804.01327.
* **Code References**: [`../include/sovsolve/solver/Iis.hpp`](../include/sovsolve/solver/Iis.hpp), [`../src/solver/Iis.cpp`](../src/solver/Iis.cpp).
* **Part Used**: **Theorem 3.4**: IIS index sets for $Ax \le b$ are exactly the support sets of the vertices of the alternative polyhedron $P = \{ y : A^T y = 0, b^T y = -1, y \ge 0 \}$.

---

## 7. Primal Heuristics for MIP

* **Diploma Thesis**: Berthold, T. (2006). *"Primal Heuristics for Mixed Integer Programs"*. Diploma Thesis, TU Berlin / ZIB. (Cited as **[B]**)
* **Code References**: [`../src/solver/MilpSolve.cpp`](../src/solver/MilpSolve.cpp#L985), [`../include/sovsolve/model/Options.hpp`](../include/sovsolve/model/Options.hpp#L353-L372).
* **Sections Used**: Alg. 3 / §3.1.2 (Objective Feasibility Pump with "FP095" parameterization), Alg. 4 / §3.2.1 (Relaxation Enforced Neighborhood Search - RENS).

---

## 8. Cutting Plane Separators & Dual Bounds

* **MIP Cutting Planes Thesis**: Wolter, K. (2006). *"Implementation of Cutting Plane Separators for Mixed Integer Programs"*. Diploma Thesis, TU Berlin. (Cited as **[W]**)
  * **Code References**: [`../include/sovsolve/solver/MilpSeparators.hpp`](../include/sovsolve/solver/MilpSeparators.hpp), [`../include/sovsolve/model/Options.hpp`](../include/sovsolve/model/Options.hpp#L415).
  * **Sections Used**: §6.1 (GMI numerical safeguards, `MAXROUNDS = 15`), Complemented Mixed Integer Rounding (c-MIR) cuts.
* **Exact Dual Bounds Paper**: Steffy, D. E., & Wolter, K. (2013). *"Valid linear programming bounds for exact mixed-integer programming"*. *Mathematical Programming Computation*, 5(3), 241–267.
  * **Code References**: [`../include/sovsolve/solver/DualBound.hpp`](../include/sovsolve/solver/DualBound.hpp), [`../src/solver/DualBound.cpp`](../src/solver/DualBound.cpp).
  * **Section Used**: §2 (Neumaier-Shcherbina primal-bound-shift for safe dual bounds).
* **c-MIR Cuts Paper**: Marchand, H., & Wolsey, L. A. (2001). *"Aggregation and mixed integer rounding inequalities"*. *Mathematical Programming*, 91(2), 289–322.
  * **Code Reference**: [`../include/sovsolve/solver/MilpSeparators.hpp`](../include/sovsolve/solver/MilpSeparators.hpp#L10).

---

## 9. Interior-Point Methods & Predictor-Corrector Loops

* **Mehrotra Predictor-Corrector**: Mehrotra, S. (1992). *"On the implementation of a primal-dual interior point method"*. *SIAM Journal on Optimization*, 2(4), 575–601.
  * **Code References**: [`FORMULATION.md`](FORMULATION.md#L981), [`../src/solver/gpu/PredictorCorrector.cu`](../src/solver/gpu/PredictorCorrector.cu#L575).
  * **Part Used**: Affine scaling direction, centering parameter $\sigma = (\mu_{\text{aff}}/\mu)^3$, second-order Taylor cross-term correction ($dx_{\text{aff}} \cdot dz_{\text{aff}}$, etc.).
* **Gondzio Centrality Correctors**: Gondzio, J. (1996). *"Multiple centrality correctors in primal-dual interior point methods"*. *Computational Optimization and Applications*, 6(2), 137–156.
  * **Code References**: [`../src/solver/gpu/PredictorCorrector.cu`](../src/solver/gpu/PredictorCorrector.cu#L412), [`../include/sovsolve/model/Options.hpp`](../include/sovsolve/model/Options.hpp#L69).
  * **Part Used**: Multiple centrality corrector steps (`apply_gondzio_correctors`).
* **Primal-Dual Regularization**: Altman, A., & Gondzio, J. (1999). *"Regularized symmetric indefinite systems in interior point methods for linear programming"*.
  * **Code Reference**: [`ARCHITECTURE-REVIEW.md`](ARCHITECTURE-REVIEW.md#L175).

---

## 10. Matrix Scaling & Equilibration

* **Ruiz $L_\infty$ Equilibration**: Ruiz, D. (2001). *"A scaling algorithm to equalize row and column norms"*. *RAL Technical Report RAL-TR-2001-034*.
* **Pock-Chambolle Pass**: Chambolle, A., & Pock, T. (2011). *"A first-order primal-dual algorithm for convex problems with applications to imaging"*. *Journal of Mathematical Imaging and Vision*, 40(1), 120–145.
* **Code References**: [`../src/solver/Scaler.cpp`](../src/solver/Scaler.cpp#L43-L88), [`../include/sovsolve/model/Options.hpp`](../include/sovsolve/model/Options.hpp#L692-L716).
* **Part Used**: Ruiz $L_\infty$-norm equilibration (iterating until row/col norms converge to 1) followed by one Pock-Chambolle $L_2$ pass.

---

## 11. Basis LU Factorization & Basis Updates

* **Markowitz Ordering & Safeguards**: Markowitz, H. M. (1957). *"The elimination form of the inverse and its application to linear programming"*. *Management Science*, 3(3), 255–269.
  * **Code References**: [`../include/sovsolve/solver/simplex/LuFactor.hpp`](../include/sovsolve/solver/simplex/LuFactor.hpp#L4), [`../src/solver/simplex/LuFactor.cpp`](../src/solver/simplex/LuFactor.cpp).
  * **Part Used**: Markowitz count $(r_i-1)(c_j-1)$ with threshold partial pivoting; Markowitz threshold $|a_{ij}| \ge 0.01 \max |a_{\cdot j}|$.
* **Product-Form of Inverse (PFI)**: Dantzig, G. B., & Orchard-Hays, W. (1953). *"The product-form for the inverse in the simplex method"*. *Mathematical Tables and Other Aids to Computation*, 7(42), 67–76.
  * **Code References**: [`../include/sovsolve/solver/simplex/LuFactor.hpp`](../include/sovsolve/solver/simplex/LuFactor.hpp#L5), [`HIGHS-COMPARISON.md`](HIGHS-COMPARISON.md#L60).

---

## 12. General Presolve & Ratio Tests

* **Presolving in Linear Programming**: Andersen, E. D., & Andersen, K. D. (1995). *"Presolving in linear programming"*. *Mathematical Programming*, 71(2), 221–245.
  * **Code References**: [`HIGHS-COMPARISON.md`](HIGHS-COMPARISON.md#L18), [`../src/solver/Presolver.cpp`](../src/solver/Presolver.cpp).
  * **Part Used**: Algebraic derivations for empty rows/cols, free-column singletons, singleton-row tightening, duplicate column merging, and general bounded-singleton elimination.
* **Harris Ratio Test**: Harris, P. M. J. (1973). *"Pivot selection methods of the Devex LP code"*. *Mathematical Programming*, 5(1), 1–28.
  * **Code Reference**: [`SIH-VERSION-PROGRESS.md`](SIH-VERSION-PROGRESS.md#L279).

---

## 13. Conflict Analysis & Reliability Branching

* **MIP Conflict Analysis**: Witzig, J., Berthold, T., & Heinz, S. (2017). *"Experiments with Conflict Analysis in Mixed Integer Programming"*. *Operations Research Proceedings 2016*, 161–167 / ZIB-Report 17-08.
  * **Code Reference**: [`../include/sovsolve/model/Options.hpp`](../include/sovsolve/model/Options.hpp#L434).
* **Reliability Branching & Pseudocosts**: Achterberg, T., Koch, T., & Martin, A. (2005). *"Branching rules revisited"*. *Operations Research Letters*, 33(1), 42–54. (Cited as **[AKM]**)
  * **Code Reference**: [`../include/sovsolve/solver/MilpSolve.hpp`](../include/sovsolve/solver/MilpSolve.hpp#L26-L31).
* **GPU Presolve Literature**: Preprint arXiv:2609.16182.
  * **Code Reference**: [`../include/sovsolve/solver/MilpSolve.hpp`](../include/sovsolve/solver/MilpSolve.hpp#L175).

---
