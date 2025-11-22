# KidneyTumorCNN
# Housing Price Prediction

A simple **Linear Regression** project to predict house prices using the Ames Housing dataset from Kaggle.

### What this project does:
- Takes house features like number of bedrooms, bathrooms, living area, year built, garage, etc.
- Cleans the data (removes nulls, handles categorical variables, drops highly correlated columns)
- Creates useful new features:
  - House Age (YearSold – YearBuilt)
  - Total Bathrooms (full + half + basement)
  - House Type (based on building type + garage size)
- Plots graphs to see how different features affect price
- Trains a Linear Regression model
- Predicts house sale price
- Achieves ~76% accuracy on test data

Built in Python using:
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Jupyter Notebook

Just a basic, clean, beginner-friendly machine learning project from 2019. No fancy stuff, no GPU needed.
