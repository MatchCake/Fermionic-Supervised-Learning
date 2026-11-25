# A fermionic framework for quantum machine learning benchmarks

[![arXiv](https://img.shields.io/badge/arXiv-2611.31458-b31b1b.svg)](https://arxiv.org/abs/2611.31458)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Code accompanying the paper

> Jérémie Gince, Ayana Sarkar, and Stefanos Kourtis,
> *A fermionic framework for quantum machine learning benchmarks*,
> [arXiv:2611.31458](https://arxiv.org/abs/2611.31458) (2026).

## Abstract

We present a general framework for the construction and benchmarking of scalable machine learning (ML) models based on the simulation of free-fermion systems.
We demonstrate that our fermionic ML (FermiML) framework is (i) versatile enough to encompass a wide range of ML techniques, including kernel methods and neural networks, (ii) applicable to common ML tasks, including classification and regression, and (iii) scalable to real-world datasets. FermiML models are quantum circuits built out of the same building blocks as most quantum ML (QML) models, namely, (parameterized) qubit rotations and entangling gates. FermiML thus allows us to build fermionic surrogates that mimic the architecture – and hence the quantum resources and inductive bias – of any given QML model, while remaining efficiently simulable classically. This, in turn, enables us to perform equal-footing benchmarks of QML models against close fermionic counterparts on any dataset and to ascertain whether any purported performance gain in QML is truly due to classically inaccessible resources or simply an accidental side-effect. To demonstrate this, we comprehensively benchmark quantum kitchen sinks (QKS), instantaneous quantum polynomial (IQP) and generic parameterized quantum circuit (PQC)-based QML models against fermionic surrogates across five classification datasets comprising 31 configurations. We find that fQKS significantly outperforms QKS and remains statistically indistinguishable from IQP and PQC, while fPQC outperforms IQP and QKS and remains statistically indistinguishable from PQC, tying it in 28 configurations. Moving beyond classification, we introduce fermionic fidelity and Pfaffian covariance kernels for continuous-target regression and establish baselines for QML in this harder task. Finally, we introduce a fermionic energy-based classifier scalable to large text (IMDb, AG News) and image (CIFAR2, Fashion) datasets, and find that it outperforms prominent QML models on image data, while beating or closely trailing them on text data and remaining close to classical ML baselines on all datasets tested. Overall, our results show that while performance depends sensitively on model architecture, the FermiML framework can be used to build fermionic surrogates that challenge or even outperform their unrestricted QML counterparts and design fermionic QML models that match the scale of classical ML baselines and reach real-world datasets.

## Repository structure

Each sub-project is a standalone Python package with its own `README.md`, dependencies, and tests.

| Directory | Paper content |
|-----------|---------------|
| [`FermiML-Framework-Classification`](FermiML-Framework-Classification) | Classification benchmarks of QKS, IQP, and PQC models against their fermionic surrogates (fQKS, fIQP, fPQC) on five datasets (31 configurations). |
| [`FermiML-Framework-Regression`](FermiML-Framework-Regression) | Fermionic fidelity and Pfaffian covariance kernels for continuous-target regression, with classical GP and SVR baselines. |
| [`FermiHamML`](FermiHamML) | Fermionic energy-based classifier on large text (IMDb, AG News) and image (CIFAR2, Fashion) datasets. |

See each sub-project's `README.md` for installation and usage.

## Citation

If you use this code, please cite:

```bibtex
@misc{gince2026fermionic,
  title         = {A fermionic framework for quantum machine learning benchmarks},
  author        = {Gince, J{\'e}r{\'e}mie and Sarkar, Ayana and Kourtis, Stefanos},
  year          = {2026},
  eprint        = {2611.31458},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.str-el},
  doi           = {10.48550/arXiv.2611.31458},
  url           = {https://arxiv.org/abs/2611.31458},
}
```

## License

This project is licensed under the Apache License 2.0; see [`LICENSE`](LICENSE).
