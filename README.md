# Performance Analysis of Hybrid and Adaptive PSO Algorithms for Multi-dimensional Data Clustering 🚀

[![Python Version](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/badge/last%20updated-November%202024-orange.svg)]()

> An advanced implementation and analysis of Particle Swarm Optimization (PSO) variants for clustering, demonstrating superior performance through hybrid and adaptive approaches.

## 📊 Key Findings

- **Superior Reliability**: APSO variants achieved 100% success rates vs basic PSO's 48.00 ± 30.00%
- **Enhanced Stability**: Hybrid variants demonstrated up to 40% better stability
- **Improved Accuracy**: Average clustering accuracy improved by 30% across all datasets
- **Efficient Processing**: Reduced computation time from 3 hours to 1.5 hours for high-dimensional data while maintaining 85%+ accuracy

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
git clone https://github.com/yourusername/DS_RP_Part-B_ClusteringAnalysis.git

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

## 📚 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{dali2024performance,
  title={Performance Analysis of Hybrid and Adaptive PSO Algorithms for Multi-dimensional Data Clustering},
  author={Dali, Shivam V},
  institution={University of Adelaide},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Steps to contribute:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Shivam V Dali - your.email@example.com

Project Link: [https://github.com/yourusername/DS_RP_Part-B_ClusteringAnalysis](https://github.com/yourusername/DS_RP_Part-B_ClusteringAnalysis)

## 🙏 Acknowledgments

* School of Mathematical Sciences, University of Adelaide
* Project Supervisor: Indu Bala
* All contributors who have helped with testing and feedback

---
⭐️ From [Shivam V Dali](https://github.com/yourusername)
