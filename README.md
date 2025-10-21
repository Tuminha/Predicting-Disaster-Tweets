# 🚨 Natural Language Processing with Disaster Tweets

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Status](https://img.shields.io/badge/Status-Learning-yellow.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)

**Learning NLP through PyTorch: Classify disaster tweets from scratch**

[🎯 Overview](#-project-overview) • [📊 Competition](#-kaggle-competition) • [🚀 Quick-Start](#-quick-start) • [📚 Learning Path](#-learning-path)

</div>

> **Learning Philosophy**: No copy-paste code allowed! Build fundamental understanding through hands-on implementation, then enhance with modern tools.

---

## 👨‍💻 Author
<div align="center">

**Francisco Teixeira Barbosa**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-black?style=flat&logo=github)](https://github.com/Tuminha)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/franciscotbarbosa)
[![Email](https://img.shields.io/badge/Email-cisco%40periospot.com-blue?style=flat&logo=gmail)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco__research-1DA1F2?style=flat&logo=twitter)](https://twitter.com/cisco_research)

*Learning Machine Learning through hands-on projects • Building NLP expertise step by step*

</div>

---

## 🎯 Project Overview

**What**: Build a text classification model to identify whether tweets are about real disasters or not using PyTorch from scratch.

**Why**: Master fundamental NLP concepts through hands-on implementation before leveraging modern transformer libraries.

**Expected Outcome**: A working disaster tweet classifier with strong baseline performance, plus deep understanding of text processing, embeddings, and neural network architectures.

### 🎓 Learning Objectives
- Master text preprocessing and tokenization fundamentals
- Build custom PyTorch models for NLP tasks
- Understand word embeddings and sequence modeling
- Learn proper train/validation/test splits for NLP
- Implement data loaders and training loops
- Evaluate model performance with appropriate metrics
- Progress from raw PyTorch to HuggingFace transformers

### 🏆 Key Achievements
- [x] Complete data exploration and EDA ✅
- [x] Implement text preprocessing pipeline ✅
- [x] Build vocabulary and custom data loaders ✅
- [x] Create PyTorch baseline model ✅
- [x] Train and evaluate model properly ✅ **80% Validation Accuracy!**
- [x] Generate Kaggle submission ✅ **0.78516 F1-Score | Rank #658**
- [ ] Enhance with HuggingFace transformers

---

## 📊 Kaggle Competition

**Source**: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

**Objective**: Predict which tweets are about real disasters (target=1) vs. not (target=0)

**Dataset**: ~7,600 training tweets with disaster labels, ~3,200 test tweets for submission

**Evaluation**: F1-Score (binary classification)

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn jupyter
# For Phase 2: transformers datasets tokenizers
```

### Setup
```bash
cd disaster_tweets
jupyter notebook
# Start with 00_exploration.ipynb
```

### Data Setup
1. Download data from Kaggle competition page
2. Place `train.csv` and `test.csv` in `data/raw/`
3. Never modify raw data - all preprocessing goes in `data/interim/`

---

## 📚 Learning Path

### Phase 1: PyTorch Fundamentals 🧠
Build everything from scratch to understand the foundations:

1. **00_exploration.ipynb** - Data exploration and EDA
2. **01_preprocessing.ipynb** - Text cleaning and tokenization
3. **02_vocab_and_dataloader.ipynb** - Vocabulary building and data loading
4. **03_model_baseline.ipynb** - PyTorch model architecture
5. **04_training_and_eval.ipynb** - Training loop and evaluation
6. **05_submission.ipynb** - Kaggle submission generation

### Phase 2: Transformers Enhancement 🚀
Enhance with modern NLP tools:

- HuggingFace transformers integration
- Pre-trained embeddings (BERT, RoBERTa)
- Advanced tokenization strategies
- Transfer learning approaches

---

## 📁 Repository Structure

```
disaster_tweets/
├── README.md                    # This file
├── data/
│   ├── raw/                    # Untouched Kaggle CSVs
│   ├── interim/               # Cleaned/preprocessed data (train_cleaned.csv, test_cleaned.csv)
│   └── processed/             # Model-ready artifacts (vocab_dict.pkl, vocab_info.json)
├── notebooks/
│   ├── 00_exploration.ipynb   # Data exploration (TODO-based)
│   ├── 01_preprocessing.ipynb # Text cleaning (TODO-based)
│   ├── 02_vocab_and_dataloader.ipynb # Data pipeline (TODO-based)
│   ├── 03_model_baseline.ipynb # Model architecture (TODO-based)
│   ├── 04_training_and_eval.ipynb # Training loop (TODO-based)
│   ├── 05_submission.ipynb    # Kaggle submission (TODO-based)
│   └── 99_lab_notes.ipynb     # Learning log and insights
├── src/
│   ├── utils/                 # Utility functions (empty)
│   └── models/                # Model definitions
│       └── baseline_model.py  # DisasterTweetClassifier implementation
└── images/                    # Plots and visualizations
```

---

## 🎯 Milestone Progress

### Phase 1: PyTorch Fundamentals ✅ **COMPLETE!**
- [x] **Exploration**: Complete EDA and understand data distribution
- [x] **Preprocessing**: Build robust text cleaning pipeline
- [x] **Data Pipeline**: Create vocabulary and efficient data loaders
- [x] **Model Architecture**: Design and implement PyTorch classifier
- [x] **Training**: Implement training loop with proper validation
- [x] **Evaluation**: Calculate metrics and analyze performance
- [x] **Submission**: Generate valid Kaggle submission file (F1: 0.785, Rank #658)

### Phase 2: Transformers Enhancement
- [ ] **HuggingFace Integration**: Replace custom components with transformers
- [ ] **Pre-trained Models**: Experiment with BERT/RoBERTa
- [ ] **Advanced Techniques**: Try different tokenization strategies
- [ ] **Performance Optimization**: Fine-tune for better F1-score

---

## 📝 Learning Documentation

**Use `99_lab_notes.ipynb` to document:**
- Key insights from each notebook
- Challenges encountered and solutions
- Performance metrics and improvements
- Ideas for future enhancements
- Questions and research directions

**Philosophy**: Every experiment should be logged. Every mistake is a learning opportunity.

---

## 📊 Key Findings from Exploration & Preprocessing

Based on the comprehensive EDA and preprocessing completed in `00_exploration.ipynb` and `01_preprocessing.ipynb`:

### Dataset Characteristics
- **Size**: 7,613 training tweets
- **Balance**: 43% disaster vs 57% non-disaster (slightly imbalanced)
- **Missing Data**: 61 missing keywords, 2,533 missing locations (33%)

### Text Patterns
- **Average tweet length**: ~100 characters
- **Average word count**: ~15 words per tweet
- **Social media features**: URLs (0.62 avg), hashtags (0.45 avg), mentions (0.36 avg)

### Preprocessing Pipeline Results
- **Domain-aware stopword removal**: Preserved 50 critical disaster-related keywords
- **Comprehensive text cleaning**: URL removal, mention/hashtag handling, emoji conversion
- **Smart text normalization**: Number-to-text conversion, whitespace normalization
- **Feature engineering**: Boolean flags for social media elements (hashtags, mentions, URLs)
- **BERT comparison**: Custom preprocessing produces 4.3x fewer tokens than BERT tokenizer

### Vocabulary Insights
- **Disaster tweets**: High frequency of words like "fire", "suicide", "disaster", "police", "killed"
- **Non-disaster tweets**: More general words like "like", "new", "get", "one", "body"
- **Clear linguistic patterns** distinguish the two classes

### 🛠 Technical Achievements

**Preprocessing Pipeline Excellence:**
- ✅ **Smart Emoji Handling**: Converts emojis to meaningful text (🔥 → "fire")
- ✅ **Domain-Aware Processing**: Preserves disaster keywords while removing noise
- ✅ **Comprehensive Cleaning**: Handles URLs, mentions, hashtags, punctuation
- ✅ **Feature Engineering**: Creates boolean flags for social media elements
- ✅ **BERT Comparison**: Demonstrates 4.3x efficiency advantage over transformer tokenization

**Data Quality Improvements:**
- **Clean datasets** saved to `data/interim/` folder
- **Consistent preprocessing** applied to both train and test sets
- **Validation pipeline** ensures data integrity and column consistency
- **Performance tracking** with word count differences for quality assurance

**Vocabulary Building Progress:**
- **Dataset Analysis**: Identified 14,644 unique words in cleaned training data
- **Data Quality Debugging**: Discovered preprocessing pipeline edge cases and UNK token handling
- **Encoding Issues**: Detected and analyzed non-ASCII characters in vocabulary
- **Preprocessing Validation**: Tested and refined text cleaning functions with comprehensive test cases

**Text-to-Sequence Conversion:**
- **Vocabulary Structure**: Implemented proper special token placement (<PAD> at index 0, <UNK> at index 1)
- **Sequence Conversion**: Created function to convert text to numerical indices with padding and truncation
- **Unknown Word Handling**: Implemented fallback mechanism for words not in vocabulary
- **Sequence Normalization**: Added padding to ensure consistent sequence lengths for batch processing

**PyTorch Data Pipeline:** ✅ **COMPLETE**
- **Custom Dataset Class**: Implemented DisasterTweetsDataset inheriting from torch.utils.data.Dataset
- **Data Loading**: Created DataLoader with batching, shuffling, and collate function
- **Tensor Conversion**: Proper conversion of text sequences and labels to PyTorch tensors
- **Batch Processing**: Efficient handling of variable-length sequences with padding
- **Vocabulary Export**: Saved vocabulary dictionary and metadata for reuse across notebooks
  - `data/processed/vocab_dict.pkl` - Word to index mapping (14,890 words)
  - `data/processed/vocab_info.json` - Metadata (vocab size, special tokens, max length)
- **Validation Results**:
  - Batch shape: `torch.Size([32, 50])` ✅
  - Labels shape: `torch.Size([32])` ✅
  - Processing speed: 0.008 seconds per batch ✅
  - Data type: `torch.int64` ✅
- **Status**: Pipeline fully operational and ready for model training! 🚀

**PyTorch Model Architecture:** ✅ **COMPLETE**
- **Model Class**: DisasterTweetClassifier (inheriting from nn.Module)
- **Architecture Design**:
  - **Embedding Layer**: Converts word indices to 100-dimensional dense vectors (3,160 vocab × 100 dims = 316,000 params)
  - **Mean Pooling**: Aggregates variable-length sequences to fixed-size representation
  - **Hidden Layer**: Fully connected layer (100 → 128) with ReLU activation
  - **Dropout**: 0.5 dropout rate for regularization
  - **Output Layer**: Binary classification head (128 → 1)
- **Model Statistics**:
  - Total trainable parameters: **329,057**
  - Input shape: `[batch_size, seq_length]` (e.g., `[32, 50]`)
  - Output shape: `[batch_size, 1]` (raw logits)
- **Training Configuration**:
  - Loss function: `BCEWithLogitsLoss()` (numerically stable sigmoid + BCE)
  - Optimizer: Adam (lr=0.001, default betas)
  - Scheduler: StepLR (step_size=5, gamma=0.1)
- **Model Modularity**: 
  - Saved to `src/models/baseline_model.py` for reuse across notebooks
  - Proper Python path configuration for importing from notebooks
- **Validation**:
  - ✅ Forward pass tested with dummy data
  - ✅ Loss calculation working correctly
  - ✅ Gradients computed successfully
  - ✅ Model works with real batches from DataLoader
- **Status**: Model architecture complete and ready for training! 🎯

---

## 🏋️ **Model Training & Results** ✅ **COMPLETE**

### **Final Model Performance: 80% Validation Accuracy! 🎯**

After systematic experimentation and hyperparameter tuning, achieved **80% validation accuracy** with excellent generalization:

| Metric | Final Result | Status |
|--------|-------------|--------|
| **Validation Accuracy** | **80.0%** | 🎯 Target achieved! |
| **F1-Score (Disaster)** | **0.76** | Strong performance |
| **Precision (Disaster)** | **78%** | High confidence predictions |
| **Recall (Disaster)** | **74%** | Catching most disasters |
| **Train/Val Gap** | **13%** | Healthy generalization |

### **The Experimental Journey: From 72% → 80%**

**Model Evolution Through 4 Iterations:**

| Iteration | Architecture | Val Acc | F1-Score | Key Learning |
|-----------|-------------|---------|----------|--------------|
| **Baseline** | vocab=3K, emb=100, hidden=128 | 72.0% | 0.68 | Severe overfitting (26% gap) |
| **V2** | vocab=41K, emb=50, hidden=64 | 77.7% | 0.74 | Better architecture |
| **V3** | + LR=1e-5, weight_decay=1e-4 | 79.4% | 0.75 | Perfect regularization |
| **Final** | + 6 epochs (early stopping) | **80.0%** | **0.76** | **Target achieved!** 🎉 |

### **Key Discoveries:**

1. **Vocabulary Size Matters** 📚
   - Increasing from 3,160 → 41,400 words captured critical disaster-specific terms
   - Rare words like "tsunami", "wildfire" are essential for classification

2. **Smaller Networks Generalize Better** 🎯
   - Reduced embedding_dim (100 → 50) and hidden_dim (128 → 64)
   - Less capacity to memorize = better generalization

3. **Learning Rate is Critical** 🐌
   - Lowering from 0.001 → 0.00001 (100× slower) enabled careful learning
   - Prevented overfitting while maintaining accuracy

4. **Multiple Regularization Techniques Stack** 🧱
   - Dropout (0.6) + Weight Decay (1e-4) + Early Stopping worked together
   - Each contributed to preventing overfitting

5. **Early Stopping Saves You** ⏹️
   - Best performance at Epoch 5 (79.8%)
   - Training beyond that point increased overfitting without gains

### **Final Model Configuration:**

```python
# Optimized Hyperparameters
vocab_size = 41,400        # Large vocabulary for rare disaster terms
embedding_dim = 50         # Compact embeddings prevent memorization
hidden_dim = 64            # Simple architecture for generalization
dropout = 0.6              # Strong regularization
learning_rate = 1e-5       # Slow, careful learning
weight_decay = 1e-4        # L2 regularization
num_epochs = 6             # Early stopping at peak performance
batch_size = 32            # Standard batch size
```

### **Training Dynamics:**

```
Epoch 1: Train 55.2% | Val 57.3%  ← Slow start
Epoch 2: Train 63.8% | Val 72.6%  ← Fast learning
Epoch 3: Train 78.1% | Val 77.0%  ← Convergence
Epoch 4: Train 85.7% | Val 79.4%  ← Optimal point
Epoch 5: Train 90.8% | Val 79.8%  ← Peak validation! 🎯
Epoch 6: Train 93.0% | Val 80.0%  ← Plateau reached
```

### **Confusion Matrix Analysis:**

**Final Model Predictions (Validation Set):**

```
                    PREDICTED
                Not Disaster  |  Disaster
        ----------------------------------------
ACTUAL  Not Disaster  |   734    |    135      (84% correct)
        Disaster      |   173    |    481      (74% correct)
```

**Real-World Impact:**
- ✅ **734 correct non-disaster identifications** (84% of 869)
- ✅ **481 disasters correctly detected** (74% of 654)
- ⚠️ **173 missed disasters** (26% false negative rate)
- ⚠️ **135 false alarms** (16% false positive rate)

**Compared to first model:**
- 40 fewer missed disasters (-19%)
- 83 fewer false alarms (-38%)
- Overall improvement across all metrics!

### **Classification Report:**

```
              precision    recall  f1-score   support

Not Disaster     0.81      0.84      0.83       869
    Disaster     0.78      0.74      0.76       654

    accuracy                           0.80      1523
   macro avg       0.80      0.79      0.79      1523
weighted avg       0.80      0.80      0.80      1523
```

**Balanced Performance:**
- Both classes perform well (F1: 0.83 and 0.76)
- No significant bias toward majority class
- Model works reliably for both disaster and non-disaster tweets

### 🖼 **Training Visualizations**

<div align="center">

<img src="images/loss and accuracy curves 6 epochs.png" alt="Training progress showing loss and accuracy curves over 6 epochs" width="800" />

*Loss and accuracy curves showing healthy convergence with minimal overfitting*

<br /><br />

<img src="images/cf 6 epochs.png" alt="Confusion matrix showing final model predictions on validation set" width="600" />

*Confusion matrix: 80% overall accuracy with balanced performance across classes*

</div>

### **🎓 Lessons Learned:**

1. **Overfitting Diagnosis:**
   - Train/val accuracy gap is the key metric
   - 25%+ gap = severe overfitting
   - 5-15% gap = healthy learning
   - <5% gap = potentially underfitting

2. **Hyperparameter Tuning:**
   - Start with learning rate (biggest impact)
   - Then regularization (dropout, weight decay)
   - Finally architecture (embedding/hidden dims)
   - Each iteration should improve validation metrics

3. **The Bias-Variance Trade-off:**
   - Large models = high variance (overfit)
   - Small models = high bias (underfit)
   - Sweet spot = 80% accuracy with 13% gap

4. **When to Stop:**
   - Monitor validation loss
   - Stop when it plateaus or increases
   - Don't chase training accuracy!

---

## 🏆 **Kaggle Competition Results** ✅ **SUBMITTED!**

### **First Submission - Baseline Model Performance**

After training and optimizing the model, submitted predictions to Kaggle competition:

<div align="center">

<img src="images/kaggle_competition_leaderboard.png" alt="Kaggle leaderboard showing first submission results" width="800" />

*First submission to Kaggle - Welcome to the leaderboard!*

</div>

### **Competition Metrics:**

| Metric | Score | Status |
|--------|-------|--------|
| **Public F1-Score** | **0.78516** | ✅ Submitted |
| **Leaderboard Position** | **#658** | First entry! |
| **Validation F1-Score** | 0.76 | Close match |
| **Difference** | +0.026 | Good generalization! |

### **Analysis:**

**🎯 Validation vs Test Performance:**
- Validation F1: **0.76**
- Public Test F1: **0.78516**
- **+2.6% improvement on test set!**

This is actually a **positive sign** - the model generalized slightly better to test data than validation data, suggesting:
- ✅ No overfitting to training set
- ✅ Robust preprocessing pipeline
- ✅ Good hyperparameter choices
- ✅ Model learned generalizable patterns

**📊 What This Score Means:**
- **Top 60% of competition** (658 out of ~1,000+ participants)
- **Solid baseline** for a from-scratch PyTorch implementation
- **Better than many pre-trained models** used without tuning
- **Room for improvement** with advanced techniques (Transformers, ensembles)

### **🎓 Learning Reflection:**

This first submission represents:
1. ✅ **Complete end-to-end ML pipeline** - From raw data to Kaggle submission
2. ✅ **Systematic experimentation** - 4 iterations improving from 72% → 80% validation
3. ✅ **Professional workflow** - Proper validation, testing, and submission process
4. ✅ **Fundamental understanding** - Built everything from scratch, no black boxes

> *It ain't much, but it's honest work.* 🚜
> 
> *— A learning journey from scratch, one tweet at a time*

### **🚀 Path to Top Performance:**

**Current Position:** #658 (0.785 F1)  
**Target:** Top 10% (~0.84 F1)  
**Gap to Close:** ~6-7% F1-Score improvement

**Potential Improvements:**

1. **Low-hanging fruit** (+2-3% F1):
   - Try LSTM/GRU instead of mean pooling
   - Experiment with max pooling
   - Ensemble multiple models
   - Fine-tune decision threshold

2. **Medium effort** (+3-5% F1):
   - Pre-trained embeddings (GloVe, FastText)
   - Data augmentation (back-translation, synonym replacement)
   - Cross-validation for robust model selection

3. **Advanced techniques** (+5-8% F1):
   - **Transformers (BERT, RoBERTa, DistilBERT)**
   - Multi-task learning
   - Pseudo-labeling
   - Model ensembles

### **📈 Competition Insights:**

**What worked:**
- ✅ Large vocabulary (41K words) captured rare disaster terms
- ✅ Aggressive regularization (dropout 0.6, weight decay) prevented overfitting
- ✅ Low learning rate (1e-5) enabled careful convergence
- ✅ Early stopping (6 epochs) avoided overtraining

**What could be better:**
- ⚠️ Simple architecture (mean pooling) doesn't capture word order
- ⚠️ No pre-trained knowledge (embeddings trained from scratch)
- ⚠️ Single model (no ensemble benefits)
- ⚠️ Fixed threshold (0.5) not optimized for F1-score

### **🎯 Next Steps:**

**Phase 2: Transformers** 🚀
- Implement HuggingFace DistilBERT/RoBERTa
- Expected gain: +5-7% F1-Score
- Target: Top 20% (0.82-0.83 F1)

**Phase 3: Advanced Techniques** 💡
- Ensemble multiple models
- Optimize decision threshold for F1
- Expected gain: +2-3% F1-Score
- Target: Top 10% (0.84+ F1)

---

### 🖼 **Exploration Visualizations**
<div align="center">

<img src="images/Tweet Length Distribution.png" alt="Tweet Length Distribution showing character count patterns" width="680" />

<br /><br />

<img src="images/Word Count Distribution.png" alt="Word Count Distribution showing vocabulary patterns" width="680" />

<br /><br />

<img src="images/Sentence Count Distribution.png" alt="Sentence Count Distribution showing text structure patterns" width="680" />

<br /><br />

<img src="images/wordclouds.png" alt="Word Clouds comparing disaster vs non-disaster tweet vocabulary" width="680" />

</div>

---

## 🛠 Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Processing | Pandas, NumPy | ETL & feature work |
| Visualization | Matplotlib, Seaborn | EDA & performance plots |
| Deep Learning | PyTorch | Model implementation |
| Evaluation | Scikit-learn | Metrics & validation |
| NLP (Phase 2) | HuggingFace | Transformers & tokenization |
| Development | Jupyter | Interactive learning |

---

## 📝 Learning Journey

**Completed Skills:**
- **Data Exploration**: Comprehensive EDA with statistical analysis and visualizations
- **Text Preprocessing**: Domain-aware cleaning pipeline with emoji handling and feature engineering
- **NLP Fundamentals**: Understanding of tokenization, stopword removal, and text normalization
- **Code Quality**: Professional implementation with error handling and comprehensive testing
- **Comparative Analysis**: Custom vs. transformer tokenization evaluation
- **Vocabulary Building**: ✅ Vocabulary creation with proper special token handling and frequency analysis
- **Text-to-Sequence Conversion**: ✅ Text to numerical indices with padding and truncation
- **PyTorch Data Pipeline**: ✅ Complete data pipeline with Dataset, DataLoader, and batching
- **Neural Network Architecture**: ✅ Designed and implemented custom PyTorch model for text classification
- **Word Embeddings**: ✅ Understanding of dense embeddings vs. one-hot encoding
- **Loss Functions & Optimizers**: ✅ Configured BCE loss and Adam optimizer for binary classification
- **Model Debugging**: ✅ Tested forward/backward passes, verified gradients and shapes
- **Training Loops**: ✅ Implemented complete training pipeline with proper validation
- **Hyperparameter Tuning**: ✅ Systematic experimentation and optimization (4 iterations, 72% → 80%)
- **Regularization Techniques**: ✅ Applied dropout, weight decay, and early stopping effectively
- **Overfitting Diagnosis**: ✅ Identified and fixed overfitting through multiple techniques
- **Performance Evaluation**: ✅ Comprehensive metrics analysis (precision, recall, F1, confusion matrix)
- **Training Dynamics**: ✅ Understanding loss curves, accuracy plots, and convergence patterns
- **Bias-Variance Trade-off**: ✅ Balanced model complexity with generalization capability

**Achievements:**
- 🎯 **80% Validation Accuracy** (from 72% baseline)
- 🎯 **0.76 F1-Score** for disaster classification
- 🎯 **13% Train/Val Gap** (healthy generalization)
- 🎯 **4 Successful Iterations** with systematic improvements

**Ready for Kaggle Submission!** 🚀

---

## 🚀 Next Steps

After completing this project:
- [ ] Try other NLP competitions on Kaggle
- [ ] Explore different model architectures (LSTM, GRU, Transformer)
- [ ] Experiment with data augmentation techniques
- [ ] Build a production API for tweet classification
- [ ] Study advanced NLP topics (attention mechanisms, transfer learning)

---

## 📄 License
MIT License - Learning project for educational purposes

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**  
*Building NLP expertise one tweet at a time* 🚀

</div>
