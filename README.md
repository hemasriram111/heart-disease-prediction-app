### **README for Heart Disease Prediction Web App**

---

#### **Project Name**: **Heart Disease Prediction Web App**

---

### **Overview**
This project is a **Heart Disease Prediction Web App** built using **Streamlit**, a popular Python library for creating web applications. The app uses a **RandomForestClassifier** machine learning model trained on the **Heart Disease Dataset** to predict whether a patient has heart disease based on their health metrics. The app also provides visualizations to compare the patient's data with the dataset.

---

### **Features**
1. **User Input**: Patients can input their health metrics using sliders in the sidebar.
2. **Prediction**: The app predicts whether the patient has heart disease or not.
3. **Visualizations**: Interactive graphs compare the patient's data with the dataset.
4. **Accuracy**: Displays the accuracy of the model on the test dataset.

---

### **Technologies Used**
- **Python**: Primary programming language.
- **Streamlit**: For building the web app.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For machine learning (RandomForestClassifier).
- **Matplotlib** and **Seaborn**: For data visualization.
- **NumPy**: For numerical computations.

---

### **Dataset**
The dataset used is the **Heart Disease Dataset**, which contains the following features:
- `age`: Age of the patient.
- `sex`: Gender (0 = Female, 1 = Male).
- `cp`: Chest pain type (0-3).
- `trestbps`: Resting blood pressure (mm Hg).
- `chol`: Serum cholesterol (mg/dl).
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False).
- `restecg`: Resting electrocardiographic results (0-2).
- `thalach`: Maximum heart rate achieved.
- `exang`: Exercise-induced angina (1 = Yes, 0 = No).
- `oldpeak`: ST depression induced by exercise relative to rest.
- `slope`: Slope of the peak exercise ST segment.
- `ca`: Number of major vessels colored by fluoroscopy.
- `thal`: Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect).
- `target`: Target variable (0 = No Heart Disease, 1 = Heart Disease).

---

### **How It Works**
1. **Data Loading**: The dataset is loaded using Pandas.
2. **Model Training**: The dataset is split into training and testing sets, and a RandomForestClassifier is trained.
3. **User Input**: The user inputs their health metrics using sliders in the sidebar.
4. **Prediction**: The trained model predicts whether the user has heart disease or not.
5. **Visualization**: Graphs are generated to compare the user's data with the dataset.
6. **Output**: The app displays the prediction and model accuracy.

---

### **Installation Steps**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction-app.git
   cd heart-disease-pred
