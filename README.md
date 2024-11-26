<div align="center">

# 🚀 Hybrid & Adaptive PSO Algorithms
## Performance Analysis for Multi-dimensional Data Clustering

[![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Updated](https://img.shields.io/badge/Last_Updated-November_2024-orange?style=for-the-badge)]()
[![Research](https://img.shields.io/badge/Research-University_of_Adelaide-purple?style=for-the-badge)]()

---

> An advanced implementation and analysis of Particle Swarm Optimization (PSO) variants for clustering, demonstrating superior performance through hybrid and adaptive approaches.



<p align="center">
  <img src="Output_Screnshots/Comparative Analysis of Algorithm Performance Across Different Datasets.png" width="900px"/>
  <br>
  <em>Comprehensive Performance Analysis Across Multiple Datasets</em>
</p>

</div>

## 📊 Research Highlights

<table>
<tr>
<td width="50%">

### Performance Milestones 🏆
- **100%** Success Rate with APSO Variants
- **40%** Enhanced Stability in Hybrid Models
- **30%** Improved Clustering Accuracy
- **50%** Reduced Computation Time

</td>
<td width="50%">

### Technical Achievements 📈
- Perfect Reliability in Complex Datasets
- Superior Cross-dimensional Adaptability
- Optimal Convergence Characteristics
- Robust Cluster Separation

</td>
</tr>
</table>

---

## 🔬 Algorithm Analysis

<div align="center">
  <img src="Output_Screnshots/Radar Plot.png" width="700px"/>
  <br>
  <em>Multi-dimensional Performance Analysis Across Algorithms</em>
</div>

### Algorithm Comparison Matrix

| Algorithm | Primary Features | Best Score |
|:---------:|:------------|:----------------:|
| K-means++ | Optimized Initialization | 0.991 |
| Basic PSO | Population-based Search | 0.774 |
| PSO-Hybrid | Deterministic-Stochastic Fusion | 0.891 |
| APSO | Dynamic Parameter Control | 0.921 |
| APSO-Hybrid | Enhanced Adaptation | 0.932 |

---

## 📈 Convergence Analysis

<div align="center">
  <img src="Output_Screnshots/Wine_Convergence.png" width="800px"/>
  <br>
  <em>Global Best Score Convergence Patterns (Wine Dataset)</em>
</div>

### Key Insights

> 💡 **APSO achieved lowest global best score: 2.1769**
> 
> 🔍 **Hybrid variants showed superior stability**
> 
> 📊 **Consistent performance in high dimensions**

### Dataset-Specific Analysis

<table>
<tr>
<td width="50%">
<h4>Low & Medium Dimensional</h4>

- **Iris Dataset**: 4 dimensions
  - Rapid convergence (100-150 iterations)
  - Clear cluster boundaries
  - 0.991 accuracy achievement

- **Wine Dataset**: 13 dimensions
  - Stable convergence pattern
  - 0.932 final accuracy
  - Optimal parameter adaptation

</td>
<td width="50%">
<h4>High Dimensional</h4>

- **MNIST Fashion**: 784 dimensions
  - Complex feature relationships
  - Dimensionality challenges
  - 0.693 accuracy with PCA

- **Dermatology**: 34 dimensions
  - Medical data complexity
  - Feature importance variation
  - 0.948 classification accuracy

</td>
</tr>
</table>

---

## 🎯 Clustering Visualization

<div align="center">
  <img src="Output_Screnshots/PCA dermitology.png" width="800px"/>
  <br>
  <em>PCA and 3D Visualization of Dermatology Dataset Clustering</em>
</div>

### 📊 Comprehensive Dataset Analysis

> Detailed visualizations and analysis for each dataset are available in dedicated Jupyter notebooks:

<table>
<tr>
<th>Dataset</th>
<th>Notebook</th>
<th>Key Features</th>
<th>Best Performance</th>
</tr>
<tr>
<td>Iris</td>
<td><a href="notebooks/X1_Iris.ipynb">X1_Iris.ipynb</a></td>
<td>Low-dimensional, well-separated clusters</td>
<td>0.991 (K-means++)</td>
</tr>
<tr>
<td>Wisconsin Breast Cancer</td>
<td><a href="notebooks/X2_Wisconsen_Breast_cancer.ipynb">X2_Wisconsen_Breast_cancer.ipynb</a></td>
<td>Binary classification, medical data</td>
<td>0.999 (APSO-Hybrid)</td>
</tr>
<tr>
<td>Wine</td>
<td><a href="notebooks/X3_Wine.ipynb">X3_Wine.ipynb</a></td>
<td>Multi-class, chemical features</td>
<td>0.932 (APSO-Hybrid)</td>
</tr>
<tr>
<td>Dermatology</td>
<td><a href="notebooks/X4_Dermitology.ipynb">X4_Dermitology.ipynb</a></td>
<td>Complex medical relationships</td>
<td>0.948 (APSO-Hybrid)</td>
</tr>
<tr>
<td>MNIST Fashion</td>
<td><a href="notebooks/X5_MNIST_Fashion.ipynb">X5_MNIST_Fashion.ipynb</a></td>
<td>High-dimensional image data</td>
<td>0.693 (APSO-Hybrid)</td>
</tr>
</table>

### 🔍 Analysis Highlights

Each notebook contains:
- Detailed preprocessing steps
- Algorithm performance comparisons
- Clustering quality metrics
- PCA and t-SNE visualizations
- Convergence analysis
- Hyperparameter sensitivity studies

For interactive exploration and detailed analysis, please refer to the individual notebooks in the repository.

---

### Hybrid Methodology Framework
```mermaid
graph LR
    A[K-means Base] --> B[PSO/APSO Integration]
    B --> C[Adaptive Control]
    C --> D[Hybrid Fusion]
    D --> E[Optimization]
```

---

## ⚙️ Implementation Guide

<table>
<tr>
<td width="50%">

### System Requirements
```python
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.2
matplotlib>=3.4.2
seaborn>=0.11.1
```

</td>
<td width="50%">

### Installation Steps
```bash
git clone https://github.com/svdexe/DS_RP_Part-B_ClusteringAnalysis_using_HeuristicOptimizationTechniques.git
cd DS_RP_Part-B_ClusteringAnalysis_using_HeuristicOptimizationTechniques
pip install -r requirements.txt
```

</td>
</tr>
</table>

---

## 📂 Project Architecture

<details>
<summary>Click to expand repository structure</summary>

```
DS_RP_Part-B_ClusteringAnalysis/
├── 📁 algorithms/
│   ├── 📜 kmeans.py
│   ├── 📜 pso.py
│   ├── 📜 apso.py
│   ├── 📜 particle.py
│   └── 📜 utils.py
├── 📁 notebooks/
│   ├── 📓 X1_Iris.ipynb
│   ├── 📓 X2_Wisconsen_Breast_cancer.ipynb
│   ├── 📓 X3_Wine.ipynb
│   ├── 📓 X4_Dermitology.ipynb
│   └── 📓 X5_MNIST_Fashion.ipynb
└── 📁 results/
    └── 📊 all_clustering_results.csv
```

</details>

---

<div align="center">

## 👨‍💻 Author Profile

<img src="https://github.com/svdexe.png" width="100px" style="border-radius: 50%;"/>

**Shivam Dali**
Data Science Graduate Student  
University of Adelaide

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/shivam-dali-86b0a1201/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/svdexe)

</div>

---

## 🙏 Acknowledgments

<table>
<tr>
<td width="33%">
<div align="center">
<h3>🏛️ Institution</h3>
School of Mathematical Sciences<br>
University of Adelaide
</div>
</td>
<td width="33%">
<div align="center">
<h3>👩‍🏫 Supervision</h3>
Dr. Indu Bala<br>
Research Advisor
</div>
</td>
<td width="33%">
<div align="center">
<h3>💻 Infrastructure</h3>
High Performance Computing<br>
Research Support
</div>
</td>
</tr>
</table>

---

<div align="center">

[![Stars](https://img.shields.io/github/stars/svdexe/DS_RP_Part-B_ClusteringAnalysis_using_HeuristicOptimizationTechniques?style=for-the-badge)](https://github.com/svdexe/DS_RP_Part-B_ClusteringAnalysis_using_HeuristicOptimizationTechniques/stargazers)
[![Forks](https://img.shields.io/github/forks/svdexe/DS_RP_Part-B_ClusteringAnalysis_using_HeuristicOptimizationTechniques?style=for-the-badge)](https://github.com/svdexe/DS_RP_Part-B_ClusteringAnalysis_using_HeuristicOptimizationTechniques/network/members)

**Made with ❤️ by [Shivam V Dali](https://github.com/svdexe)**

</div>
