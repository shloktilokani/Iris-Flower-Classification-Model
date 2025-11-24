# 🌼 **Iris Flower Classification — Machine Learning Project**

![Iris Flower Classification](res/video.gif)

---

## 🔍 **Project Overview**

This project uses the **Iris Flower Dataset** to build, train, and evaluate two supervised machine learning models:

- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**

The goal is to classify iris species based on flower measurements.

---

## 🌱 **Nature of Dataset**

The **Iris dataset** contains **150 samples**, each with **4 numerical features**:

- 🌸 *Sepal Length*
- 🌸 *Sepal Width*
- 🌸 *Petal Length*
- 🌸 *Petal Width*

🎯 **Target Variable:** `species`

The dataset contains **3 classes**:

- *Iris-setosa*
- *Iris-versicolor*
- *Iris-virginica*

---

## ⚙️ **Project Workflow**

### **1️⃣ Import Libraries**

- `pandas` → Data handling
- `sklearn` → ML models & metrics

---

### **2️⃣ Load Dataset**

```python

df = pd.read_csv('IRIS.csv')
target_column = 'species'

```

Load iris data into a Pandas DataFrame and define the target column.

---

### **3️⃣ Data Preparation**

Split the dataset into:

- **Features (X)** → Flower measurements
- **Target (y)** → Species

```python

X = df.drop(target_column, axis=1)
y = df[target_column]

```

---

### **4️⃣ Train-Test Split**

Dataset split:

- **40% Training Data**
- **60% Testing Data**

```python

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.6, random_state=20
)

```

---

### **5️⃣ Model Initialization & Training**

#### **🔹 K-Nearest Neighbors (KNN)**

- Parameter used: `n_neighbors = 25`

```python

knn_model = KNeighborsClassifier(n_neighbors=25)
knn_model.fit(X_train, y_train)

```

#### **🔹 Decision Tree Classifier**

- Parameters: `max_depth = 3`, `random_state = 42`

```python

tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

```

---

### **6️⃣ Model Prediction**

Predict using both models:

```python

knn_predictions = knn_model.predict(X_test)
tree_predictions = tree_model.predict(X_test)

```

---

### **7️⃣ Model Evaluation**

#### **📊 Accuracy Scores**

- **KNN Accuracy:** *88.89%*
- **Decision Tree Accuracy:** *87.78%*

```python

knn_accuracy = accuracy_score(y_test, knn_predictions)
tree_accuracy = accuracy_score(y_test, tree_predictions)

```

📌 KNN performed slightly better than the Decision Tree.

---

## 🧪 **8️⃣ Custom Data Testing**

Three categories of test cases were created:

- ✔️ *Easy Cases*
- ⚠️ *Ambiguous Cases*
- ❗ *Outlier Cases*

Each case contains custom measurements and predictions from both models were compared.

---

## 💡 **Insights**

- **KNN** provided marginally better accuracy with a higher neighbor value (`25`).
- **Decision Tree** performed well but showed slight instability for ambiguous test cases.
- **Outlier predictions** varied significantly between the two models.
- The dataset is clean and well-balanced, requiring minimal preprocessing.

---

## 🏁 **Conclusion**

Both the **KNN** and **Decision Tree** models successfully classified the Iris species with high accuracy (>87%).

📌 **KNN (25 neighbors)** proved to be the best performing model for this dataset.

This project demonstrates the complete ML workflow: data loading, preprocessing, model training, evaluation, and custom testing.

---

## 📦 **Project Files**

- `IRIS.csv` — Dataset
- `Avash_Shlok.pdf` — Project report/working
- Python code snippets included in README

---

## 🚀 **How to Run the Project**

1. Install dependencies:

   ```bash
   pip install pandas scikit-learn
   ```

2. Place `IRIS.csv` in the working directory
3. Run the Python script or notebook
4. View model accuracy and predictions

---

## 👥 Contributors

We are a team of five members who collaborated on this data analysis project. Connect with us:

- **Shlok Tilokani**  
  - [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/shloktilokani)  
  - [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shloktilokani/)

- **Avash Sahu**  
  - [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/avashsahu)  
  - [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/avashsahu/)

---

## ⭐ **Thank You!**

Feel free to ⭐ star the repo if you found this helpful!
