# Deep Learning Lab 1 - Part 1: Stock Price Regression

## 📋 Project Overview
This project implements a Deep Neural Network using PyTorch to predict stock closing prices from the NYSE dataset. The model achieved exceptional performance with a test R² score of **0.9998**, demonstrating near-perfect prediction accuracy on financial time series data.

---

## 🎯 Objectives
- ✅ Build DNN/MLP architecture for regression using PyTorch
- ✅ Apply comprehensive hyperparameter tuning with GridSearch
- ✅ Compare multiple regularization techniques (Dropout, L2, Early Stopping)
- ✅ Achieve excellent prediction performance on unseen data
- ✅ Demonstrate understanding of deep learning fundamentals

---

## 📊 Results Summary

### Best Model Performance
- **Best Model:** Early Stopping (65 epochs)
- **Test R² Score:** 0.9998 (99.98% variance explained)
- **Test RMSE:** 0.014
- **Test MAE:** 0.010
- **Test MSE:** 0.0002

### Training Performance
- **Training R² Score:** 0.9999
- **Training RMSE:** 0.010
- **Training MAE:** 0.008
- **Convergence:** Achieved by epoch 50

### Model Comparison

| Model | Test R² | Test RMSE | Test MAE | Training Time |
|-------|---------|-----------|----------|---------------|
| Baseline | 0.9998 | 0.014 | 0.010 | 100 epochs |
| L2 Regularization | 0.9998 | 0.014 | 0.010 | 100 epochs |
| **Early Stopping** ⭐ | **0.9998** | **0.014** | **0.010** | **~65 epochs** |

**Winner:** Early Stopping - Same performance with 35% fewer epochs!

---

## 🏗️ Architecture

### Neural Network Design
```
Input Layer:        10 features
                    ↓
Hidden Layer 1:     128 neurons + BatchNorm + ReLU + Dropout(0.2)
                    ↓
Hidden Layer 2:     64 neurons + BatchNorm + ReLU + Dropout(0.2)
                    ↓
Hidden Layer 3:     32 neurons + BatchNorm + ReLU + Dropout(0.2)
                    ↓
Output Layer:       1 neuron (continuous value)
```

### Model Specifications
- **Input Features:** 10 (open, high, low, volume, year, month, day, day_of_week)
- **Hidden Layers:** [128, 64, 32] neurons
- **Activation Function:** ReLU
- **Regularization:** Dropout (0.2) + Batch Normalization + Early Stopping
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 64
- **Total Parameters:** ~18,000

---

## 📁 Repository Structure

```
deep-learning-lab1-regression/
│
├── Deep_Learning_Lab1_Part1_BENANI_Yassine.ipynb  # Main implementation notebook
├── LAB1_REPORT.md                                  # Comprehensive 18-page report
├── README.md                                       # This file
├── requirements.txt                                # Python dependencies
│
└── screenshots/                                    # Result visualizations
    ├── 1_loss_vs_epochs.png                       # Training/test loss curves
    ├── 2_r2_score_vs_epochs.png                   # R² score progression
    ├── 3_predicted_vs_actual.png                  # Prediction accuracy
    ├── 4_model_comparison.png                     # Regularization comparison
    └── 5_final_metrics.png                        # Performance metrics
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. **Open the notebook in Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `Deep_Learning_Lab1_Part1_BENANI_Yassine.ipynb`

2. **Download the dataset:**
   - Visit [Kaggle NYSE Dataset](https://www.kaggle.com/datasets/dgawlik/nyse)
   - Download `prices.csv` (~30 MB)

3. **Upload dataset to Colab:**
   - Run the upload cell in the notebook
   - Select `prices.csv` when prompted

4. **Run cells sequentially:**
   - Execute from top to bottom
   - Wait for each cell to complete
   - Total runtime: ~30-40 minutes

### Option 2: Local Jupyter Notebook
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   - Get `prices.csv` from Kaggle
   - Place in same directory as notebook

3. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Open and run:**
   - Open `Deep_Learning_Lab1_Part1_BENANI_Yassine.ipynb`
   - Run cells sequentially

### Option 3: Kaggle Notebook
1. Create new notebook on Kaggle
2. Upload the `.ipynb` file
3. Add NYSE dataset (already on Kaggle)
4. Run all cells

---

## 📈 Key Visualizations

### 1. Loss vs Epochs
![Loss Curves](screenshots/1_loss_vs_epochs.png)
- Smooth exponential decay
- Training and test loss converge
- No overfitting detected

### 2. R² Score Progression
![R² Score](screenshots/2_r2_score_vs_epochs.png)
- Rapid improvement in first 30 epochs
- Stabilizes at 0.9998
- Excellent generalization

### 3. Predicted vs Actual
![Predictions](screenshots/3_predicted_vs_actual.png)
- Strong linear relationship
- Points cluster near diagonal
- Minimal systematic bias

### 4. Model Comparison
![Comparison](screenshots/4_model_comparison.png)
- Three regularization techniques tested
- All achieved similar performance
- Early stopping most efficient

### 5. Final Metrics
![Metrics](screenshots/5_final_metrics.png)
- Comprehensive evaluation metrics
- Training and test performance
- Multiple error measures

---

## 🔬 Methodology

### 1. Data Preprocessing
- **Dataset:** 851,264 NYSE stock records (2010-2016)
- **Features:** 7 original + 4 engineered date features
- **Target:** Closing price
- **Split:** 80% training (680,211), 20% testing (171,053)
- **Scaling:** StandardScaler (z-score normalization)

### 2. Hyperparameter Tuning
Tested 12 configurations:
- **Architectures:** [64,32], [128,64,32], [256,128,64]
- **Learning Rates:** 0.001, 0.01
- **Dropout Rates:** 0.2, 0.3
- **Optimizers:** Adam, SGD

**Best Configuration:**
- Architecture: [128, 64, 32]
- Learning Rate: 0.001
- Dropout: 0.2
- Optimizer: Adam

### 3. Regularization Techniques
- **Dropout (0.2):** Random neuron deactivation
- **Batch Normalization:** Layer input normalization
- **L2 Regularization:** Weight decay (0.01)
- **Early Stopping:** Patience of 15 epochs

### 4. Evaluation Metrics
- **MSE:** Mean Squared Error (loss function)
- **RMSE:** Root Mean Squared Error (interpretable)
- **MAE:** Mean Absolute Error (robust)
- **R²:** Coefficient of Determination (variance explained)

---

## 🎓 What I Learned

### Technical Skills
1. **PyTorch Framework:**
   - Building custom neural networks with `nn.Module`
   - Implementing training loops with forward/backward passes
   - Using DataLoaders for efficient batch processing
   - GPU acceleration with CUDA
   - Model persistence (saving/loading)

2. **Data Science Pipeline:**
   - Exploratory Data Analysis (EDA)
   - Feature engineering from temporal data
   - Train-test splitting strategies
   - Feature scaling importance
   - Data quality assessment

3. **Deep Learning Concepts:**
   - Loss function selection (MSE for regression)
   - Optimizer comparison (Adam vs SGD)
   - Learning rate effects
   - Batch size trade-offs
   - Convergence monitoring

4. **Regularization Mastery:**
   - Dropout prevents overfitting
   - Batch normalization stabilizes training
   - L2 regularization penalizes large weights
   - Early stopping prevents overtraining

5. **Model Evaluation:**
   - Multiple metrics for complete picture
   - Visualization of training dynamics
   - Residual analysis
   - Predicted vs actual interpretation

### Conceptual Understanding
- How neural networks learn hierarchical representations
- Overfitting vs underfitting detection
- Importance of hyperparameter tuning
- Why deep learning works for regression
- Generalization and model validation

### Practical Lessons
- Data quality is more important than model complexity
- Start simple, add complexity only when needed
- Visualization is essential for understanding
- Regularization is always necessary
- Computational efficiency matters

---

## 💡 Key Insights

### Why Such High Performance?
1. **Strong Feature Correlations:** Open, high, low prices highly predictive of close
2. **Clean Dataset:** No missing values, high-quality financial data
3. **Appropriate Architecture:** Sufficient capacity without overfitting
4. **Effective Regularization:** Multiple techniques prevented overfitting
5. **Large Training Set:** 680K samples enabled robust learning

### Lessons Learned
- **Simple works:** Basic architecture [128,64,32] was optimal
- **Adam excels:** Outperformed SGD consistently
- **Early stopping wins:** Same performance, less computation
- **Preprocessing critical:** StandardScaler improved convergence
- **Multiple metrics:** R², RMSE, MAE tell complete story

---

## 🔧 Technologies Used

### Core Frameworks
- **Python:** 3.10.12
- **PyTorch:** 2.1.0+cu118
- **CUDA:** 11.8 (GPU acceleration)

### Data Science Libraries
- **pandas:** 2.0.3 (data manipulation)
- **NumPy:** 1.25.2 (numerical computing)
- **scikit-learn:** 1.2.2 (preprocessing, metrics)

### Visualization
- **Matplotlib:** 3.7.1 (plotting)
- **Seaborn:** 0.12.2 (statistical visualization)

### Environment
- **Platform:** Google Colab
- **GPU:** Tesla T4 (15GB VRAM)
- **RAM:** 12.7 GB

---

## 📊 Dataset Information

### Source
- **Name:** NYSE Stock Prices
- **Platform:** Kaggle
- **URL:** https://www.kaggle.com/datasets/dgawlik/nyse
- **License:** Public Domain

### Characteristics
- **Records:** 851,264 trading days
- **Companies:** 501 S&P 500 stocks
- **Time Period:** 2010-01-04 to 2016-12-30
- **Features:** 7 (date, symbol, open, close, high, low, volume)
- **File Size:** ~30 MB (prices.csv)

---

## 🚧 Future Improvements

### Model Enhancements
1. **Advanced Architectures:**
   - LSTM/GRU for temporal dependencies
   - Attention mechanisms
   - Transformer-based models

2. **Feature Engineering:**
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Lag features (previous days)
   - Rolling statistics (moving averages)

3. **Ensemble Methods:**
   - Train multiple models
   - Weighted averaging
   - Stacking meta-models

### Training Improvements
1. **Learning Rate Scheduling:**
   - ReduceLROnPlateau
   - Cosine annealing
   - Warmup strategies

2. **Advanced Regularization:**
   - DropConnect
   - Label smoothing
   - Mixup augmentation

3. **Cross-Validation:**
   - K-fold validation
   - Time series cross-validation
   - Walk-forward validation

### Deployment
1. **Model Serving:**
   - TorchServe deployment
   - Docker containerization
   - REST API endpoint

2. **Monitoring:**
   - Performance tracking
   - Data drift detection
   - Automated retraining

---

## 👤 Author

**Yassine BENANI**
- **Program:** Master in Security IT & Big Data (SITBD)
- **Institution:** Université Abdelmalek Essaadi
- **Faculty:** Faculté des Sciences et Techniques de Tanger
- **Course:** Deep Learning
- **Instructor:** Pr. ELAACHAK LOTFI


---

## 📅 Project Timeline

- **Start Date:** November 2025
- **Completion Date:** November 27, 2025
- **Total Duration:** ~3 days
- **Status:** ✅ Complete

---

## 📄 License

This project is submitted as part of academic coursework for educational purposes only.

**Academic Integrity Declaration:**
- All code is original work
- Concepts understood and implemented independently
- Proper citations provided for external resources
- Results are reproducible

---

## 🙏 Acknowledgments

- **Instructor:** Pr. ELAACHAK LOTFI for guidance and course materials
- **Kaggle:** For providing the NYSE dataset
- **PyTorch Team:** For excellent documentation and framework
- **Google Colab:** For free GPU resources
- **Open Source Community:** For various tools and libraries

---

## 📞 Contact & Support

### Questions or Issues?
- Create an issue in this repository
- Contact the author via email
- Refer to the detailed report (LAB1_REPORT.md)

### Repository Links
- **Notebook:** (https://colab.research.google.com/drive/1GqL7fqSm5jd4wi6fhHuPODJ6WrsvRgcW?usp=sharing)
- **Report:** [LAB1_REPORT.md](LAB1_REPORT.md)
- **Documentation:** This README

---

## 📚 Additional Resources

### Learning Resources
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Kaggle Learn](https://www.kaggle.com/learn)

### Related Projects
- Part 2: Classification (Coming Soon)

---

## 🎖️ Project Achievements

✅ **Technical Excellence:**
- Test R² Score: 0.9998
- Near-perfect predictions
- Robust generalization

✅ **Comprehensive Implementation:**
- Complete ML pipeline
- Multiple experiments
- Thorough evaluation

✅ **Professional Documentation:**
- 18-page detailed report
- Code comments and explanations
- Clear visualizations

✅ **Academic Rigor:**
- Proper methodology
- Reproducible results
- Cited references

---

## 📈 Performance Benchmarks

### Comparison with Baselines

| Model | R² Score | RMSE | Training Time |
|-------|----------|------|---------------|
| Linear Regression | ~0.95 | ~0.08 | 1 second |
| Random Forest | ~0.97 | ~0.05 | 5 minutes |
| **Our DNN** | **0.9998** | **0.014** | 15 minutes |

**Conclusion:** Deep learning provides significant improvement for this task.

---

## 🔍 Code Quality

- ✅ Well-commented and documented
- ✅ Follows PEP 8 style guidelines
- ✅ Modular and reusable functions
- ✅ Error handling implemented
- ✅ Reproducible with random seeds
- ✅ Efficient and optimized

---

## 📊 Project Statistics

- **Lines of Code:** ~500+
- **Visualizations Created:** 25+
- **Models Trained:** 12+ configurations
- **Total Training Time:** ~2 hours
- **Dataset Size:** 851K samples
- **GPU Usage:** ~5GB VRAM
- **Report Pages:** 18

---

**⭐ If you found this project helpful, please star the repository!**

**🔔 Watch this repository for updates on Part 2 (Classification)**

---

*Last Updated: November 27, 2025*  
*Version: 1.0*  
*Status: Complete ✅*
