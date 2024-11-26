# Performance Analysis of Hybrid and Adaptive PSO Algorithms for Multi-dimensional Data Clustering 🚀

[![Python Version](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/badge/last%20updated-November%202024-orange.svg)]()

> An advanced implementation and analysis of Particle Swarm Optimization (PSO) variants for clustering, demonstrating superior performance through hybrid and adaptive approaches.

<div align="center">
  <img src="Output_Screnshots/Comparative Analysis of Algorithm Performance Across Different Datasets.png" width="800"/>
  <p><em>Figure 1: Performance comparison across different datasets showing superior APSO performance</em></p>
</div>

## 📊 Key Findings

- **Superior Reliability**: APSO variants achieved 100% success rates vs basic PSO's 48.00 ± 30.00%
- **Enhanced Stability**: Hybrid variants demonstrated up to 40% better stability
- **Improved Accuracy**: Average clustering accuracy improved by 30% across all datasets
- **Efficient Processing**: Reduced computation time from 3 hours to 1.5 hours for high-dimensional data while maintaining 85%+ accuracy

## 🎯 Algorithm Performance

<div align="center">
  <img src="Output_Screnshots/Radar Plot.png" width="600"/>
  <p><em>Figure 2: Radar plot showing multi-dimensional performance metrics across algorithms</em></p>
</div>

### Convergence Analysis

<div align="center">
  <img src="Output_Screnshots/Wine_Convergence.png" width="800"/>
  <p><em>Figure 3: Convergence behavior showing APSO achieving lowest global best score (2.1769)</em></p>
</div>

## 🔍 Clustering Visualization

<div align="center">
  <img src="Output_Screnshots/PCA dermitology.png" width="800"/>
  <p><em>Figure 4: 3D and PCA visualization of clustering results across different algorithms</em></p>
</div>


## 🎯 Features

- **Multiple Algorithm Implementations**:
  - K-means clustering (baseline)
  - Basic PSO
  - PSO-Hybrid
  - Adaptive PSO (APSO)
  - APSO-Hybrid

- **Comprehensive Dataset Analysis**:
  - Iris (low-dimensional)
  - Wisconsin Breast Cancer (binary classification)
  - Wine (multi-class)
  - Dermatology (complex features)
  - MNIST Fashion (high-dimensional)
 

## 💡 Visual Results Summary

Our comprehensive visual analysis demonstrates:

1. **Superior Clustering Quality**
   - Clear cluster separation in APSO-Hybrid
   - Consistent performance across dimensions
   - Robust handling of complex datasets

2. **Convergence Efficiency**
   - Faster convergence in hybrid variants
   - More stable optimization trajectory
   - Better final solution quality

3. **Multi-dimensional Excellence**
   - Balanced performance across metrics
   - Strong showing in high-dimensional cases
   - Consistent cross-dataset reliability

## 🛠️ Implementation

### Prerequisites

```python
# Required packages
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.2
matplotlib>=3.4.2
seaborn>=0.11.1
```

### Quick Start

```python
# Clone the repository
git clone https://github.com/svdexe/DS_RP_Part-B_ClusteringAnalysis_using_HeuristicOptimizationTechniques.git
# Install dependencies
pip install -r requirements.txt

# Run analysis notebooks
jupyter notebook
```

## 📁 Repository Structure

```
DS_RP_Part-B_ClusteringAnalysis/
├── algorithms/
│   ├── kmeans.py
│   ├── pso.py
│   ├── apso.py
│   ├── particle.py
│   └── utils.py
├── notebooks/
│   ├── X1_Iris.ipynb
│   ├── X2_Wisconsen_Breast_cancer.ipynb
│   ├── X3_Wine.ipynb
│   ├── X4_Dermitology.ipynb
│   └── X5_MNIST_Fashion.ipynb
├── data/
│   └── processed/
├── results/
│   └── all_clustering_results.csv
└── README.md
```

## 📈 Performance Visualization

### Algorithm Success Rate Analysis
```
Algorithm     Success Rate    Std Dev
----------------------------------------
APSO          100.00%        0.000
APSO-Hybrid   100.00%        0.000
K-means++     100.00%        0.000
PSO-Hybrid     62.00%        0.303
PSO            48.00%        0.303
```

## 🔬 Methodology

Our approach progresses through multiple algorithmic enhancements:

1. **Baseline Implementation**: K-means clustering with optimized initialization
2. **PSO Integration**: Population-based search with velocity-position updates
3. **Adaptive Mechanisms**: Dynamic parameter control (w(t), c1(t), c2(t))
4. **Hybrid Approaches**: Combined deterministic and stochastic optimization

## 📊 Key Results

- Achieved perfect reliability (100%) with APSO variants
- Demonstrated complexity-dependent performance ranging from 0.988-0.999 for structured datasets to 0.566-0.693 for complex scenarios
- Successfully reduced computational complexity while maintaining accuracy
- Established effectiveness of hybrid PSO-based clustering for complex, high-dimensional datasets

## 👨‍💻 Contributor

- Shivam Dali is a Data Science graduate student from Adelaide University. Connect with him on [LinkedIn](https://www.linkedin.com/in/shivam-dali-86b0a1201/) and explore more projects on [GitHub](https://https://github.com/svdexe).


## 🙏 Acknowledgments

* School of Mathematical Sciences, University of Adelaide
* Project Supervisor: Indu Bala
* All contributors who have helped with testing and feedback

---
⭐️ From [Shivam V Dali](https://github.com/svdexe)
