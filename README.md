# Proteinoid Spike Analysis Project

## Overview
This project implements a comprehensive analysis framework for proteinoid spike trains, focusing on their universal approximation capabilities through deep ReLU networks. The project demonstrates that proteinoid substrates, owing to their chemical makeup and proto-cognitive abilities, can be interpreted as universal approximators through a novel equivalence between their electrical activity and deep ReLU networks.

## Scientific Background

### Proteinoid Spikes
Proteinoids are soft matter fluidic systems that exhibit oscillatory electrical activity due to cationic and anionic exchange. This project analyzes their computational capabilities through:
- Voltage-sensitive dye recordings of electrical activity
- Transformation of analog signals to digital spike trains
- Complex network analysis of spike patterns
- Deep learning-based classification

### Key Theoretical Components
1. **Transformation Functions**
   - F₁ (Spiral Sampling): `x(t) = 10 + (10-2t)cos(t·π)`
   - F₂ (Significant Digit Extraction): Extracts first significant digit after decimal point

2. **Complexity Analysis**
   - Eight distinct graph-based metrics
   - Meta-metric for overall complexity evaluation
   - Network topology analysis

3. **Classification Model**
   - 16-dimensional feature vector space
   - Accuracy: 70.41%
   - Theoretical connection to Kolmogorov-Arnold representation

## Project Structure

```
project_structure/
├── README.md
├── requirements.txt
├── raw_data/                  
│   ├── .gitignore            
│   └── README.md             
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py    # Spike train data processing
│   ├── models/
│   │   ├── __init__.py
│   │   └── neural_network.py # Classification model
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── complexity.py     # Graph-based metrics
│   │   └── visualization.py  # Data visualization
│   └── utils/
│       ├── __init__.py
│       └── metrics.py        # Performance metrics
└── scripts/
    ├── train_classifier.py   # Model training
    └── analyze_complexity.py # Complexity analysis
```

## Installation

```bash
# Clone repository
git clone https://github.com/username/proteinoid-spike-analysis.git
cd proteinoid-spike-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Unix
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Processing
```python
from src.data.data_loader import load_spike_data

# Load and process spike train data
dataframes = load_spike_data("path/to/meta_data.csv")
```

### Training Classification Model
```bash
python scripts/train_classifier.py --data_path /path/to/meta_data.csv \
                                 --epochs 100 \
                                 --batch_size 16
```

### Running Complexity Analysis
```bash
python scripts/analyze_complexity.py --data_path /path/to/meta_data.csv \
                                   --output_dir /path/to/output
```

## Feature Engineering

The project uses a 16-dimensional feature vector for spike classification:
1. **Temporal Features**
   - Time (tᵢ)
   - Time difference (Δtᵢ)
   - Inter-Spike Interval (ISIᵢ)

2. **Statistical Features**
   - Coefficient of Variation of ISIs (CV_ISIᵢ)
   - Rolling means (μS,i,w for w = 3,5,10)
   - Rolling standard deviations (σS,i,w for w = 3,5,10)

3. **Spectral Features**
   - Sine transformations (Fsin,p,i for p = 5,10,20)
   - Cosine transformations (Fcos,p,i for p = 5,10,20)

## Model Architecture

```
Input Layer (16 features)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Dense Layer (16 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

## Performance Metrics

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 70.41% |
| Precision | 70.59% |
| Recall    | 55.81% |
| F1 Score  | 0.6234 |
| AUC       | 0.73   |

## Data Requirements

Input data should be a CSV file containing:
- Columns: Data1, Data2, Data3, Data4, Data5
- Sampling rate: 600 points per second
- Voltage-sensitive dye recordings
- Binary spike indicators (0 or 1)

## Legal and Attribution

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
If you use this code in your research, please cite:
```bibtex
@article{sharma2024proteinoid,
  title={Proteinoid spikes: from protocognitive to universal approximating agents},
  author={Sharma, Saksham and Mahmud, Adnan and Tarabella, Giuseppe and Mougoyannis, Panagiotis and Adamatzky, Andrew},
  journal={arXiv preprint},
  year={2024}
}
```

### Contributors
- Saksham Sharma (Unconventional Computing Laboratory, UWE Bristol)
- Adnan Mahmud (Department of Chemical Engineering, Cambridge)
- Giuseppe Tarabella (IMEM-CNR)
- Panagiotis Mougoyannis (UWE Bristol)
- Andrew Adamatzky (UWE Bristol)

## Support

For questions or issues:
1. Check existing GitHub issues
2. Create a new issue with:
   - Detailed description
   - Minimal reproducible example
   - System information
   - Error messages/logs

## Future Directions

1. **Model Improvement**
   - Enhanced feature engineering
   - Architecture optimization
   - Increased training data

2. **Theoretical Extensions**
   - Investigation of random vs. structured sequences
   - Advanced proteinoid physical generative AI models
   - Universal-computing multiverse paradigms
