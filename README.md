# Toward a Massively Parallel Recursive Relational Machine: A CUDA-Based Virtual Architecture for Logic Programming

**José Carpio Cañada**  
Department of Information Technology, University of Huelva, Spain  
Contact: jose.carpio@dti.uhu.es · [GitHub](https://github.com/jcarpio/)  
**Manuscript type:** Research concept paper and experimental architecture proposal  
**Date:** 25 October 2026  
**Status:** Conceptual proposal; no performance measurements or implementation claims are made.

## Abstract

Logic programming provides a high-level, relational description of computation, yet most widely used implementations execute on processors whose prevailing programming and execution models are instruction-oriented. This paper proposes the **Recursive Relational Dataflow Machine (RRDM)**, a virtual architecture that decomposes recursive and relational computations into fine-grained logical tasks mapped onto massively parallel devices, initially NVIDIA CUDA GPUs. Instead of assigning one physical processor to each recursive call, RRDM represents computations as dynamically activated nodes for decomposition, unification, constraint propagation, branching, and reduction. A scheduler maps potentially millions of virtual nodes onto a bounded pool of GPU execution resources. We examine a motivating example in which a factorial of 10,000 is decomposed into independently generated factors and combined by a hierarchical product reduction, contrasting the sequential dependency chain of a conventional recursive implementation with the logarithmic depth of an associative reduction tree under constant-time arithmetic assumptions. The paper explicitly separates *work*, *span*, *processor count*, *communication*, and *bit complexity*: the architecture does not make factorial, full-list traversal, or exact unification constant-time. We outline a Turing-complete, pure definite-clause relational subset, together with fair search, environment-isolated branching, and a staged GPU execution model. A falsifiable research methodology compares sequential CPU, optimized GPU reduction, a graph-runtime prototype, and existing logic-programming implementations on factorial, lists, trees, unification, nondeterministic search, and constraints. The contribution is an experimentally testable synthesis of relational semantics, dataflow activation, parallel logic programming, and GPU scheduling—not a claim of a novel asymptotic algorithm or demonstrated speedup. The long-term objective is to determine whether logic-oriented virtual instructions offer an effective foundation for future heterogeneous and specialized hardware.

**Keywords:** Prolog; logic programming; CUDA; GPU; dataflow architecture; recursion; unification; parallel reduction; AND-parallelism; OR-parallelism; virtual machine; relational programming.

---

## 1. Introduction and motivation

Declarative languages invite programmers to describe *what relationships hold* instead of prescribing a sequence of state-changing instructions. Prolog illustrates this approach through unification, recursive clauses, and goal solving. A common implementation technique is the Warren Abstract Machine (WAM), whose abstract instruction set was designed to support compiled Prolog on conventional processors and potentially in firmware or hardware [1, 2]. This success does not imply that every aspect of logical execution maps naturally to a single sequential instruction stream. The relationship between a relational program and its execution plan remains a central research opportunity.

Consider a class of list predicates that repeatedly decompose a term into head and tail, process a smaller problem, and combine results. The conceptual computational structure is a graph of dependencies among logical subproblems. A sequential runtime usually traverses this graph in a chosen operational order; however, independent goals or alternatives may admit parallel execution. Research on parallel Prolog has long distinguished **AND-parallelism** (concurrent goals) from **OR-parallelism** (concurrent search alternatives), while acknowledging difficult issues in variable bindings, memory, scheduling, and search [3, 4]. Separately, dataflow machines have investigated the principle that an operation becomes enabled when its operands are available, with dynamic token tagging used to distinguish multiple activations [5–7].

The central research question is therefore:

> **Can a relational program be compiled into a graph of fine-grained logical micro-operations whose activation and scheduling exploit GPU parallelism more effectively than a conventional instruction-centered execution strategy?**

This paper proposes **RRDM**, initially a software virtual machine implemented on CUDA rather than new silicon. Thousands or millions of *virtual* logical nodes can be hosted by a device with a much smaller number of hardware arithmetic lanes. Its novelty must be tested against decades of work in parallel logic programming, dataflow execution, GPU reductions, and graph processing; it should not be presumed from the terminology alone.

### 1.1 Motivating hypotheses

- **H1 — Expressivity:** A sufficiently expressive pure definite-clause kernel with unbounded terms and recursion can serve as a Turing-complete computational language, assuming unbounded abstract resources and suitable execution semantics.
- **H2 — Mapping:** Statically identifiable independent reductions, goal conjunctions, and finite-domain operations may benefit from batched GPU execution.
- **H3 — Granularity:** Extremely fine-grained node creation may be slower than conventional execution unless virtual nodes are fused or batched.
- **H4 — Semantics:** Fair search and branch-local variable substitutions can preserve answers of a pure relational subset independently of particular scheduling order, subject to resource limits and duplicate-answer handling.
- **H5 — Specialization:** A restricted relational kernel may allow more aggressive parallelization than full Prolog with procedural control, effects, and order-sensitive primitives.

These are **hypotheses to be evaluated**, not empirical results.

## 2. Background and related work

### 2.1 Prolog execution and the WAM

Warren's abstract Prolog instruction set established a practical architecture for unification, environments, choice points, and backtracking [1]. Aït-Kaci's reconstruction explains how these mechanisms are built and optimized [2]. RRDM does not aim to displace the WAM before establishing a compelling advantage; it proposes an alternative decomposition of execution and a comparative evaluation.

### 2.2 Parallel logic programming

The comprehensive survey by Gupta *et al.* [3] documents shared-memory OR-parallelism, AND-parallelism, combinations, analyses, and memory concerns. Dovier *et al.* revisit parallel logic programming in a later survey [4]. In RRDM, AND- and OR-parallelism are explicit runtime graph operations with different correctness and memory requirements. Calling every goal a GPU task without analyzing variable dependencies would be incorrect.

### 2.3 Dataflow and tagged tokens

Arvind and Culler survey static and dynamic dataflow, including token-tagging and data structures [5]. The tagged-token work at MIT is particularly relevant: activations become distinguishable through tags and can be scheduled on operand availability [6, 7]. RRDM borrows this *design family*, adding logic-specific notions such as terms, unification, search alternatives, and constraints. A GPU implementation is a host for the experiment, not itself a true hardware dataflow computer.

### 2.4 Parallel prefix/reduction and GPU execution

Blelloch describes parallel scans and related operations as central parallel building blocks [8], and GPU-specific efficient scan methods have been investigated [9]. NVIDIA's CUB provides thread-, warp-, block-, and device-level collective primitives, including reduction and scan [10]. CUDA's SIMT execution model groups threads into 32-thread warps and imposes costs for divergence, synchronization, and irregular memory access [11]. Therefore, CUDA cores must not be equated to fully independent autonomous microprocessors; the proposed virtual nodes are scheduled work items, not persistent one-to-one physical cores.

### 2.5 Pure relational programming and search

miniKanren research demonstrates the expressive benefits and implementation challenges of relational programming without fixed input/output directions [12]. RRDM takes inspiration from the relational viewpoint but must specify its own exact syntax, fairness properties, arithmetic relations, occurs-check policy, constraints, and operational semantics. It is not sufficient to assert that removing `cut` automatically produces an efficient or complete parallel interpreter.

## 3. Research problem and scope

We seek a mapping

\[
\mathcal{C} : (P, Q) \mapsto G(P,Q), \qquad \mathcal{E}(G, D) \mapsto \text{answer stream},
\]

where `P` is a pure relational program, `Q` a query, `G` a graph of virtual micro-operations, `D` an execution device, and `E` a scheduler/executor. Subject to well-defined fairness and finite-prefix observability, the answers must agree with the declarative semantics of `P` for the supported fragment.

**In scope:** finite and recursively structured terms; pure Horn-clause predicates; unification; conjunction/disjunction; branch-local environments; fair search; simple numeric constraints; batched GPU execution; comparison with sequential and optimized parallel baselines.

**Out of scope in the first prototype:** arbitrary Prolog I/O; `cut`; dynamic assertion/retraction; meta-level predicates with nonlogical instantiation tests; full ISO Prolog compatibility; transactional side effects; claimed O(1) whole-program complexity; physical construction of millions of independent processing units.

### 3.1 Semantic commitments

1. Program clauses are declarative; the interpreter may reorder independent goals only when the transformation is justified.
2. Substitutions belong to a logical branch, or use an equivalent persistent/versioned representation.
3. OR-branches cannot overwrite each other's bindings.
4. A fair scheduler must not starve a reachable finite proof indefinitely solely because another branch diverges, under the abstract machine's fairness assumptions.
5. Logical termination and resource exhaustion are distinct outcomes.
6. The occurs check is defined explicitly. The initial mathematically clean prototype should use finite first-order terms with occurs-check unification; rational trees may be a separate extension.
7. Arithmetic relations require a declared domain. GPU fixed-width modular integers and exact unbounded integers are **different semantics**.

## 4. Architecture of the Recursive Relational Dataflow Machine

### 4.1 Three-layer architecture

```text
   Pure relational program + query
                 |
           Front end / IR
                 |
     Dependency + mode + purity analysis
                 |
       Graph of virtual logical nodes
        /       |        |       \
     MATCH    UNIFY     SPAWN    REDUCE
        \       |        |       /
       Work queues / branch environments
                 |
      CUDA batched graph execution
                 |
      Answers, residual constraints, traces
```

The compiler and runtime are deliberately separated: the language describes relations; the intermediate representation (IR) describes logical dependencies; the scheduler decides which eligible tasks run on CPU or GPU.

### 4.2 Proposed virtual instruction set

| Instruction | Meaning | Main implementation concern |
|---|---|---|
| `MATCH(tag, term)` | Match a term constructor or literal | Irregular memory access |
| `SPLIT_LIST(term)` | Expose head and tail if a cons cell | Representation and type checks |
| `UNIFY(x,y,env)` | Unify under a branch-local substitution | Conflicts, occurs check |
| `SPAWN(goal,env)` | Create a logical activation | Task overhead and memory |
| `FORK(options,env)` | Enumerate clause/search alternatives | Exponential search width |
| `JOIN(deps,env)` | Await and combine dependent results | Synchronization |
| `GUARD(c)` | Test or propagate a constraint | Domain-specific propagation |
| `REDUCE(op,inputs)` | Combine associative values | Correct associativity and identity |
| `YIELD(answer)` | Emit a solution | Ordering/backpressure |
| `SUSPEND(var)` | Wait for needed logical information | Deadlock and fairness |

Instructions are **conceptual** rather than measured GPU kernels. For performance, a real implementation would fuse groups of operations and employ structure-of-arrays storage, compact task descriptors, and bounded queues.

### 4.3 Virtual nodes versus CUDA resources

Let `V` denote the number of outstanding virtual tasks and `P` the effective number of simultaneously usable processing lanes. Usually `V >> P` is possible because tasks occupy queues or memory while hardware processes batches. The number of advertised “CUDA cores” alone does not specify attainable concurrency, especially for integer multiplication, 64-bit arithmetic, large-integer arithmetic, or irregular control flow [11]. The initial emulator should use conventional CUDA kernels and CUB rather than attempt one recursive child-kernel launch per logical node.

### 4.4 Memory and synchronization

The critical problem is not increment/decrement but **variable identity and binding**. A plausible design comprises immutable term arenas, branch-local substitution overlays, work queues and a compact pool of constraint states. Persistent substitutions or copy-on-write branch snapshots trade extra memory for correctness. Alternative designs include union–find-like structures with versioning, but concurrent destructive union is unsafe across nondeterministic OR branches without isolation.

A scheduler should enforce memory budgets and queue backpressure and support cancellation and reproducible traces. Term sharing and locality determine whether many-node execution is viable.

## 5. Factorial as a minimal architectural experiment

### 5.1 Sequential recursive definition

```prolog
factorial(0, 1).
factorial(N, F) :-
    N #> 0,
    N1 #= N - 1,
    factorial(N1, F1),
    F #= N * F1.
```

The conventional computation has a dependency chain of length `N` when evaluated straightforwardly. Assigning each recursive frame to a different virtual node does **not** remove that dependency. If each result is needed before the predecessor completes, the critical path remains linear.

### 5.2 Re-expression as an associative product

For natural `N`, factorial is also

\[
N! = \prod_{i=1}^{N} i.
\]

For a fixed-width modular experiment (not exact factorial), the monoid `(Z / 2^w Z, multiplication, 1)` supports associative parallel reduction. An exact-integer implementation similarly uses associativity but with variable-cost multiplication. A balanced reduction tree has `ceil(log2 N)` multiplication **dependency levels** if values are available and pairwise operations can run concurrently.

For `N = 10,000`, an ideal full binary tree has **14 levels**, not two levels. A device with effective capacity for 5,000 independent unit-cost operations can perform the first reduction level in one wave (5,000 products), followed by levels with 2,500, 1,250, and so on. In the simplified model, this is at least 14 dependent levels. Preparing 10,000 independent inputs in two waves is a separate stage from combining them.

**Important:** CUDA threads are not permanently attached to each integer; a kernel can process indices in a grid, with device scheduling distributing work over available hardware. “Two batches of 5,000 CUDA cores” is a useful intuition for throughput, but not an exact GPU execution model.

### 5.3 Work–span analysis

Let `W(N)` be total work and `S(N)` critical-path depth (span). For fixed-width unit-time multiplication:

| Method | Work | Span | Comment |
|---|---:|---:|---|
| Sequential factorial recursion | Θ(N) | Θ(N) | Chain of dependent products |
| Chain mapped to virtual nodes | Θ(N) plus scheduling | Θ(N) | More nodes do not erase dependencies |
| Balanced parallel product tree | Θ(N) | Θ(log N) | Requires associative refactoring |

Under a work–span model, an idealized lower-bound benchmark for execution on `P` processors is governed by `max(W/P, S)`; upper bounds for particular schedulers require additional assumptions. Real time adds data transfer, allocation, kernel launches, synchronization and contention.

### 5.4 Exact-integer factorial and bit complexity

`10000!` has approximately **35,660 decimal digits**. This excludes native fixed-width integer types. The exact runtime must account for operand sizes and the multiplication algorithm (schoolbook, Karatsuba, FFT-family algorithms, etc.). If `M(b)` denotes the cost of multiplying `b`-bit integers, later tree levels involve larger `b`, so treating all 14 levels as 14 equal-duration clock ticks is invalid. No `O(1)` exact factorial claim follows from parallel factor generation.

A sound experimental sequence is:

1. **Modular factorial:** compare 32-bit and 64-bit results modulo `2^w`, with explicit overflow semantics. Note that for sufficiently large `N` the result is trivially zero modulo `2^w`; use additional nondegenerate associative product benchmarks so this corner case does not distort conclusions.
2. **Modular product over a prime:** choose a prime modulus larger than the tested `N` when needed, avoiding premature zero outcomes while retaining bounded arithmetic.
3. **Exact factorial:** use a rigorously verified multiprecision strategy (CPU baseline, GPU research implementation if available) and report end-to-end costs.
4. **Recursive graph execution:** run the RRDM scheduler on the same associative reduction, then compare against optimized GPU reduction to measure the overhead of the virtual-node abstraction itself.

### 5.5 A minimum useful demonstration

The first prototype should expose a trace such as:

```text
Input: factorial_product(10000)
Task kind: GENERATE_RANGE
Items: 10000
Reduce operator: multiply_mod_p
Tree depth (ideal): 14
Active virtual tasks: variable
Peak physical concurrency: measured, not assumed
Total elapsed time: measured
Correctness: compared with reference modular product
```

It would be scientifically misleading to describe this as a generic Prolog speedup until unification and true relational search are evaluated.

## 6. Recursive lists and structural decomposition

Consider the pure relation:

```prolog
append([], Ys, Ys).
append([X|Xs], Ys, [X|Zs]) :-
    append(Xs, Ys, Zs).
```

For a pointer-based immutable cons cell, exposing `head` and `tail` can be constant-time. Yet traversing an arbitrary linked list, summing all elements, or computing its length requires handling information from all relevant cells; these are not generally O(1) problems. A parallel reduction of `N` already-addressable values has logarithmic ideal depth with sufficient processing resources; an unknown pointer chain may entail a sequential discovery dependency, unless indexing, pointer-jumping, or preprocessing changes the representation and cost model.

This suggests an explicit choice between **logical lists** and **physical layouts**. A compiler might flatten immutable lists to contiguous vectors, construct auxiliary indexes, or represent a term graph with shared nodes. Such transformations have costs, semantic constraints and different suitability for queries with partially unbound terms.

Potential RRDM primitives include `SPLIT_LIST`, `MAP_REL`, `FOLD_ASSOC`, `SPLIT_TREE`, and `JOIN_REL`, with optimizations applied only when their properties are established. The IR must retain the original relational meaning while optimizing particular grounded modes.

## 7. A pure relational language kernel

### 7.1 Language design

The first dialect should resemble definite-clause Prolog:

```prolog
membero(X, [X|_]).
membero(X, [_|Xs]) :- membero(X, Xs).

appendo([], Ys, Ys).
appendo([X|Xs], Ys, [X|Zs]) :- appendo(Xs, Ys, Zs).
```

A clean core includes predicates, first-order terms, recursion, conjunction, and disjunction through multiple clauses. Constraints should be added as logically specified extensions. Effects would live in a controlled outer interface, distinct from the pure kernel.

### 7.2 Turing completeness versus operational completeness

Pure definite-clause logic programming over unbounded term structures is computationally expressive enough to encode general computation; thus a suitable kernel can be **Turing complete in the abstract mathematical sense**. This requires unbounded memory and potentially unbounded execution. Real GPUs are finite machines and are not literally unbounded.

A separate question is whether the **search strategy** fairly exposes all finite derivations. Naive depth-first execution may become trapped in an infinite branch. A fair search policy (for example, interleaving or breadth-controlled scheduling) improves solution completeness but has memory and performance tradeoffs. Turing completeness does not mean all queries terminate, and declarative equivalence does not guarantee equivalent answer order or real-world performance.

### 7.3 Search and parallel binding environments

AND-parallel goals can share logical variables. Their schedules and bindings must respect correctness. OR-parallel branches demand isolated substitutions or carefully versioned binding stores. Initial milestones should support **ground and independent subgoals first**, followed by constrained dependent goals. An unrestricted global variable store shared by thousands of threads would not be a correct Prolog execution model.

## 8. Compiler and GPU execution strategy

### 8.1 Intermediate representation

Each node descriptor could store:

```text
node_id, opcode, input_refs, output_refs,
branch_id, environment_id, dependency_count,
state, priority, continuation_id
```

Edges represent data dependencies, branch provenance, or constraint wake-up conditions. A counter reaching zero may enqueue a data-ready node. Tagged activation identifiers are important when a recursive predicate is invoked many times [5–7].

### 8.2 Static and dynamic optimizations

- **Mode/dependency analysis:** identify ground or independent subgoals.
- **Associative reduction recognition:** transform recognized folds only if the operator and identity satisfy required laws.
- **Task fusion:** aggregate small logical operations so launch/scheduling costs do not overwhelm work.
- **Batching by opcode:** group similar tasks to reduce SIMT divergence.
- **Work stealing or queue balancing:** redistribute irregular workloads, with a measured synchronization cost.
- **Branch pruning:** propagate valid logical constraints to avoid unnecessary search.
- **Hybrid execution:** move very small, irregular, or highly sequential tasks to CPU when appropriate.

### 8.3 Why CUDA is a useful testbed—and a hard one

CUDA exposes substantial throughput but is optimized for organized collections of threads. GPUs have a finite register file, shared memory, warp schedulers, and device memory bandwidth. Irregular unification and backtracking can cause branch divergence and scattered loads; work queues and dynamic activation can also dominate execution. A negative performance result on CUDA would not invalidate the abstract relational-machine model, but it would constrain claims about this implementation substrate.

## 9. Evaluation methodology

### 9.1 Research questions

- **RQ1:** For associative reductions, how much additional overhead does the RRDM abstraction introduce relative to optimized CUDA primitives?
- **RQ2:** For grounded list/tree workloads, can compilation into batches preserve correctness and improve end-to-end throughput?
- **RQ3:** At what task granularity do GPU scheduling and memory costs outweigh useful parallelism?
- **RQ4:** How does fair OR-search compare with conventional Prolog search in time-to-first-answer, answer coverage under bounded resources, and memory?
- **RQ5:** Which unification/constraint workloads are favorable for GPU acceleration, and which should remain on CPU?
- **RQ6:** How much of any observed speedup is due to **algorithmic restructuring**, rather than the proposed virtual-machine instructions?

### 9.2 Benchmark matrix

| Benchmark | Purpose | Correctness oracle | Main metrics |
|---|---|---|---|
| Modular range product / factorial | Parallel reduction proof of concept | CPU reference modulo p | Latency, throughput, launch cost |
| Exact factorial | Big-integer stress test | Trusted multiprecision library | Exactness, time, memory |
| Ground list map/sum | Regular and irregular memory layouts | CPU reference | End-to-end time, bytes transferred |
| List `append/3` modes | Relations with different instantiation patterns | Reference Prolog answers | Answer set, latency, memory |
| Tree recursion | Independent subtrees and joins | CPU reference | Speedup vs tree shape |
| First-order unification | Variable binding and structural traversal | Reference unifier with occurs check | Success/failure, binding equivalence |
| Nondeterministic search | OR-parallelism and fairness | Bounded exhaustive reference | Time to first/k answers, coverage |
| Constraint propagation | Wake-up and pruning behavior | Trusted CLP(FD) reference subset | Propagations/s, consistency |

### 9.3 Baselines and fairness of comparison

Compare (A) direct sequential CPU C++ for algorithmic primitives, (B) a mature conventional Prolog implementation for relational predicates, (C) optimized native CUDA/CUB algorithms, (D) CPU RRDM interpreter, and (E) CUDA RRDM. Where possible, report warm and cold runs separately. Avoid comparing an unoptimized Prolog predicate only against an optimized GPU product tree and then attributing the full advantage to architecture.

For every device record GPU model, SM count, memory size, CUDA toolkit version, compiler flags, CPU model, system memory and software versions. Benchmark repetition, variability, clock states and synchronization points must be described. GPU timings require explicit synchronization for asynchronous operations; total wall time should include data transfers if an application would incur them.

### 9.4 Measurements

Report speedup `T_reference / T_RRDM` with confidence intervals or variability distributions, along with:

- Total work units processed; estimated and measured critical path where possible.
- Active vs queued virtual tasks and peak GPU occupancy.
- Memory allocation, bytes moved, host↔device transfer volume.
- Time to first solution, time to `k` solutions, and completeness under a budget.
- Correctness comparisons of answer substitutions, not merely counts.
- Failure modes: exhausted memory, deadlocks, starvation, numerical overflow, incorrect branch interference.

### 9.5 Ablation studies

Run the same workload with and without task fusion, opcode batching, branch-local substitution optimizations, mode analysis, reduced queue traffic, and CPU/GPU partitioning. This is necessary to isolate which architectural ideas actually matter.

## 10. Expected challenges and threats to validity

1. **Critical paths remain:** parallel hardware cannot erase true data dependencies.
2. **Asymptotic claims are model-dependent:** O(1) local operations do not imply O(1) complete answers.
3. **Exact arithmetic scales with bit length:** huge factorials are not constant-cost multiplications.
4. **Unification is irregular:** pointer-heavy term graphs may perform poorly on GPUs.
5. **Binding semantics are subtle:** incorrect sharing of substitutions can silently corrupt logical answers.
6. **Nondeterminism causes combinatorial growth:** more workers do not guarantee economical exploration.
7. **GPU architecture is SIMT, not arbitrary independent agents:** warps, blocks, divergence and synchronization matter [11].
8. **Transfer and kernel overhead matter:** small goals may be slower on GPU.
9. **Baseline selection matters:** product-tree speedups are not themselves novel, and reductions are already highly optimized [8–10].
10. **Adoption arguments are speculative:** Prolog's prevalence cannot be explained solely by learning difficulty or a mismatch with von Neumann machines without separate empirical study.
11. **Power and energy costs matter:** reporting only throughput could hide inefficient acceleration.
12. **Finite memory and fairness:** a mathematically fair abstract scheduler may not guarantee practical progress when queues overflow.

## 11. Staged implementation roadmap

**Phase 0 — Specification and reference semantics.** Define pure term syntax, substitutions, unification with occurs check, answer equivalence, and the reference scheduler. Establish reference test suites.

**Phase 1 — Factorial and reduction primitives.** Implement modular products and balanced reductions using CUDA/CUB. Explicitly demonstrate why two waves of factor generation differ from full reduction; establish native GPU baselines.

**Phase 2 — CPU graph virtual machine.** Implement descriptors, readiness counters, continuations, branch environments and reproducible instrumentation. Validate results before GPU parallelism.

**Phase 3 — CUDA micro-task executor.** Port selected regular primitives, use batched opcode queues, compare against CUB and CPU. Measure fusion thresholds.

**Phase 4 — Structural recursion.** Compile ground list and tree predicates into executable graphs. Evaluate list layout transformations and performance boundaries.

**Phase 5 — Relational semantics.** Add unification, branch-local alternatives, fair search and selected constraints. Compare answer sets with mature logic engines.

**Phase 6 — Hardware co-design study.** Only if data justify it, identify high-value primitives for FPGA or specialized accelerators, such as term matching, queue scheduling, or branch-local unification.

### 11.1 Reproducibility deliverables

A public research repository should include the language/IR specification, reference interpreter, CUDA kernels, test cases, workload generators, benchmark harness, exact hardware/software manifests, raw results, analysis scripts, and a license. Results should distinguish **demonstrated findings** from goals and predictions.

## 12. Discussion: implications for logic-programming architecture

An important conceptual outcome is the distinction between **recursive syntax**, **logical dependency graphs**, and **physical scheduling**. The best hardware realization of a recursive definition is not necessarily a hardware chain mimicking recursive calls. Some programs admit parallel associative reformulations; others contain unavoidable sequential dependencies; still others are dominated by unification or search. RRDM is intended to make these structures explicit so a scheduler can choose among CPU, GPU, and eventually dedicated hardware.

Thus, the central scientific novelty to test is not “factorial in 14 steps,” a known implication of balanced reduction. It is whether **a reusable, semantics-preserving relational graph IR and runtime** can discover, represent, and efficiently schedule worthwhile parallelism across a meaningful class of logic programs with less application-specific rewriting.

A fully compatible Prolog runtime and a smaller pure relational dialect are both legitimate futures. The former offers easier adoption but more operational constraints; the latter offers freedom to reorder and execute, but requires deliberate definitions of search, arithmetic, effects, and answer behavior. The proposed phased approach postpones this decision until experiments reveal which restrictions actually help.

## 13. Conclusion

This paper proposes a CUDA-hosted Recursive Relational Dataflow Machine in which potentially millions of virtual logic micro-operations are dynamically scheduled across finite hardware resources. The architecture draws on established research in WAM-based Prolog, parallel logic programming, tagged-token dataflow, and GPU reduction, and proposes their integration into a testable relational virtual-machine stack. A factorial-of-10,000 thought experiment illustrates both the appeal of massive parallelism and its limits: factor generation can be batched, while full combination requires reduction levels and exact computation incurs substantial bit complexity. The proposal therefore rejects unconditional O(1) claims and instead emphasizes correctness, work–span reasoning, scheduling overhead, and rigorous comparisons.

The next decisive step is not specialized silicon but an open reference implementation and a reproducible benchmark suite. A successful result would identify classes of relational computations for which a data-ready logical execution model offers practical benefits. An unsuccessful result would still clarify the limits of GPU-supported micro-task execution and inform the design of future logic-oriented architectures.

---

## References

[1] Warren, D. H. D. (1983). *An Abstract Prolog Instruction Set*. Technical Note 309, SRI International. https://www.sri.com/publication/cyber-formal-methods-pubs/an-abstract-prolog-instruction-set/

[2] Aït-Kaci, H. (1991). *Warren's Abstract Machine: A Tutorial Reconstruction*. MIT Press. https://mitpress.mit.edu/9780262510585/warrens-abstract-machine/

[3] Gupta, G., Pontelli, E., Ali, K. A. M., Carlsson, M., & Hermenegildo, M. V. (2001). Parallel Execution of Prolog Programs: A Survey. *ACM Transactions on Programming Languages and Systems*, 23(4), 472–602. https://doi.org/10.1145/504083.504085

[4] Dovier, A., Formisano, A., Gupta, G., Hermenegildo, M. V., Pontelli, E., & Rocha, R. (2022). Parallel Logic Programming: A Sequel. *Theory and Practice of Logic Programming*. https://www.cambridge.org/core/journals/theory-and-practice-of-logic-programming/article/abs/parallel-logic-programming-a-sequel/594EDA0899ED5AC1F86E3E5A3D6431C8

[5] Arvind & Culler, D. E. (1986). Dataflow Architectures. *Annual Review of Computer Science*, 1, 225–253. https://doi.org/10.1146/annurev.cs.01.060186.001301

[6] Culler, D. E. (1985). *Resource Management for the Tagged Token Dataflow Architecture*. MIT-LCS-TR-332. https://hdl.handle.net/1721.1/149603

[7] Traub, K. R. (1986). *A Compiler for the MIT Tagged-token Dataflow Architecture*. MIT-LCS-TR-370. https://hdl.handle.net/1721.1/149638

[8] Blelloch, G. E. (1990). *Prefix Sums and Their Applications*. CMU-CS-90-190. https://www.cs.cmu.edu/~guyb/publications.html

[9] Sengupta, S., Harris, M., & Garland, M. (2008). *Efficient Parallel Scan Algorithms for GPUs*. NVIDIA Research. https://research.nvidia.com/sites/default/files/publications/nvr-2008-003.pdf

[10] NVIDIA. *CUDA Core Compute Libraries (CCCL): CUB Documentation*. https://nvidia.github.io/cccl/unstable/cub/index.html ; https://nvidia.github.io/cccl/unstable/cub/developer_overview.html

[11] NVIDIA. *CUDA Programming Guide*, especially SIMT and hardware multithreading. https://docs.nvidia.com/cuda/cuda-programming-guide/

[12] Byrd, W. E. (2010). *Relational Programming in miniKanren: Techniques, Applications, and Implementations*. PhD dissertation, Indiana University. https://hdl.handle.net/2022/8777

### Suggested additional reading

- Arvind & Iannucci, R. A. (1985). *Two Fundamental Issues in Multiprocessing: The Dataflow Solution*. MIT-LCS-TM-241. https://hdl.handle.net/1721.1/149051
- Maa, G. K. (1988). *Code-mapping Policies for the Tagged-token Dataflow Architecture*. MIT-LCS-TR-425. https://hdl.handle.net/1721.1/149677
- Software Preservation Group. *Prolog and Logic Programming Historical Sources Archive*. https://www.softwarepreservation.org/projects/prolog/index.html

---

## Appendix A. Minimal testable pseudocode

**Associative product tree (bounded/modular arithmetic)**

```text
function product_tree(values, operation, identity):
    current = values
    if current is empty:
        return identity
    while length(current) > 1:
        launch_parallel_for j in [0, floor(length(current)/2)):
            next[j] = operation(current[2*j], current[2*j+1])
        if length(current) is odd:
            append current[last] to next
        synchronize
        current = next
    return current[0]
```

**RRDM task scheduling (concept only)**

```text
ready_queue <- nodes whose dependencies are satisfied
while ready_queue not empty or active_tasks > 0:
    batch <- dequeue compatible ready nodes
    execute batch on selected device
    for each completed node:
        write branch-local outputs
        decrement successor dependency counters
        enqueue newly ready successors
        emit validated solutions
```

The scheduler pseudocode is deliberately incomplete: termination detection, fairness, branch isolation, queue overflow, allocation failures, and CUDA synchronization require formal design and testing.

## Appendix B. Candidate falsification criteria

The design should be reconsidered if (i) virtual-node overhead systematically dominates useful work at practical sizes, (ii) branch-local unification cannot be made correct within feasible memory constraints, (iii) irregular term workloads experience consistently unacceptable SIMT divergence, or (iv) benefits appear only after manually converting relational programs into conventional data-parallel algorithms without a reusable compiler transformation. Reporting such negative results would still constitute a useful research contribution.
