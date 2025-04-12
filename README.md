# 🔥 Algerian Forest Fire Prediction

This project utilizes machine learning techniques to predict the occurrence of forest fires in Algeria based on meteorological and environmental data. By analyzing key factors such as temperature, humidity, wind speed, and rainfall, the model aims to assist in early detection and prevention of wildfires.

## 📊 Dataset

The dataset comprises observations from two regions in Algeria: Bejaia and Sidi Bel-Abbes, collected between June and September 2012. Key features include:

- **Temperature (°C)**: Noon temperature ranging from 22 to 42°C
- **Relative Humidity (%)**: Values between 21% and 90%
- **Wind Speed (km/h)**: Ranging from 6 to 29 km/h
- **Rainfall (mm)**: Daily totals from 0 to 16.8 mm
- **Fire Weather Index (FWI) Components**:
  - FFMC: 28.6 to 92.5
  - DMC: 1.1 to 65.9
  - DC: 7 to 220.4
  - ISI: 0 to 18.5
  - BUI: 1.1 to 68
  - FWI: 0 to 31.1
- **Classes**: Binary classification indicating 'fire' or 'not fire'

*Source*: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset)

## 🧠 Methodology

1. **Data Preprocessing**:
   - Handled missing values and outliers
   - Encoded categorical variables
   - Normalized numerical features

2. **Exploratory Data Analysis (EDA)**:
   - Visualized feature distributions
   - Analyzed correlations between variables

3. **Model Development**:
   - Implemented classification algorithms:
     - Logistic Regression
     - Random Forest Classifier
     - Support Vector Machine (SVM)
   - Performed hyperparameter tuning using GridSearchCV

4. **Model Evaluation**:
   - Assessed performance using:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
   - Visualized results with confusion matrices and ROC curves

## 📈 Results

The Random Forest Classifier achieved the highest performance with:

- **Accuracy**: 92%
- **Precision**: 90%
- **Recall**: 93%
- **F1-Score**: 91.5%

These results indicate a robust model capable of effectively predicting forest fire occurrences based on the provided features.

## 🚀 Future Enhancements

- **Model Optimization**: Explore advanced algorithms like XGBoost or ensemble methods
- **Feature Engineering**: Incorporate additional environmental factors for improved accuracy
- **Deployment**: Develop a web application for real-time fire risk prediction
- **Data Expansion**: Integrate more recent and diverse datasets to enhance model generalizability


## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Jupyter Notebook

## 📬 Contact

For any inquiries or collaboration opportunities, feel free to reach out:

- **Email**: pranat32@gmail.com
- **GitHub**: [Pranat1729](https://github.com/Pranat1729)
- **LinkedIn**: [linkedin.com/in/Pranat](https://www.linkedin.com/in/pranat-sharma-a55a77168/)

---

*Note*: This project is part of my ongoing efforts to apply machine learning techniques to real-world environmental challenges. Contributions and feedback are welcome!


