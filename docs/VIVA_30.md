# 30 Viva Questions and Answers (Multi-Model ML System)

1. What is the goal of this diabetes prediction project?
   - To predict the likelihood of diabetes in individuals using clinical and demographic features with supervised machine learning algorithms.

2. What algorithms are evaluated in this project?
   - Four distinct algorithms: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and Support Vector Machine (SVM).

3. What is the target variable?
   - `Outcome`: a binary classification variable (0 = Non-diabetic, 1 = Diabetic).

4. How is Logistic Regression used?
   - It acts as a linear baseline, modeling the log-odds of the positive class using the sigmoid activation function.

5. How does a Decision Tree work in this application?
   - It recursively partitions the feature space into orthogonal decision regions by selecting splits that maximize purity (minimizing Gini Impurity).

6. What is Gini Impurity?
   - A metric measuring the probability of incorrect classification of a randomly chosen element: $\text{Gini} = 1 - \sum p_i^2$. Zero indicates complete purity.

7. What is Random Forest and why is it effective?
   - An ensemble method that builds multiple decision trees using bootstrap aggregating (bagging) and random feature selection, significantly reducing tree variance and overfitting.

8. What is Support Vector Machine (SVM)?
   - A classifier that searches for the optimal hyperplane maximizing the geometric margin between the two classes.

9. What is a Kernel function in SVM?
   - A mathematical function (like the Radial Basis Function / RBF used here) that maps non-linearly separable data into a higher-dimensional space where a linear hyperplane can separate them.

10. Why is feature scaling (`StandardScaler`) essential?
    - Distance-based algorithms like SVM and gradient-based algorithms like Logistic Regression are sensitive to feature magnitudes; scaling gives each feature mean = 0 and variance = 1.

11. Why do Decision Trees not strictly require feature scaling?
    - Because decision tree splits are based on monotonic rank ordering of individual features rather than Euclidean distances.

12. Why treat zero values in glucose, blood pressure, insulin, skin thickness, and BMI as missing?
    - Because a zero measurement in these physiological variables is clinically impossible in living individuals, indicating missing data.

13. What strategy was used for imputation?
    - Median imputation (`SimpleImputer(strategy="median")`), which is robust against extreme outliers and skewed medical distributions.

14. What is a Stratified Train-Test Split?
    - Splitting the dataset (80% train / 20% test) while ensuring both subsets maintain the exact same ratio of diabetic to non-diabetic cases as the full dataset.

15. What is the difference between Accuracy and F1-Score?
    - Accuracy measures overall correct classifications out of total samples, while F1-Score is the harmonic mean of Precision and Recall, crucial for imbalanced datasets.

16. Why is Recall particularly vital in healthcare screening?
    - Recall ($\text{TP} / (\text{TP} + \text{FN})$) measures the ability to detect actual diabetic patients. A false negative could leave an individual untreated.

17. What is Precision in this context?
    - The percentage of individuals predicted as diabetic who truly are diabetic ($\text{TP} / (\text{TP} + \text{FP})$).

18. What were the benchmark results across the four models?
    - Decision Tree achieved the highest F1-score (0.6842) and test Accuracy (76.62%), with 72.22% Recall. Random Forest and SVM achieved 74.03% Accuracy, and Logistic Regression reached 70.78%.

19. What is a Confusion Matrix?
    - A 2x2 grid representing True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP).

20. How are the trained models persisted?
    - Using Python's `joblib` library to serialize the complete scikit-learn Pipeline (imputer + scaler + estimator) into `.joblib` files.

21. Why bundle preprocessing into a `Pipeline`?
    - To prevent data leakage during training and ensure identical data transformations during web inference.

22. What role does Flask play in the architecture?
    - It acts as the web application framework, receiving HTTP requests, validating inputs, dispatching model inferences, and rendering HTML templates.

23. How does the application handle database storage?
    - It supports MySQL as the primary database with an automatic SQLite fallback (`predictions.db`), ensuring zero downtime even without MySQL server setup.

24. What information is stored in the `predictions` table?
    - Patient name, chosen classifier model, 8 clinical values, predicted class (0/1), confidence probability, and submission timestamp.

25. What is the difference between Bagging and Boosting?
    - Bagging (used in Random Forest) trains independent trees in parallel on bootstrap samples; Boosting trains sequential trees where each tree corrects the errors of the previous one.

26. What is Overfitting and how do we combat it?
    - Overfitting occurs when a model memorizes training noise and fails to generalize. It is mitigated by tree depth pruning, ensembling, and cross-validation.

27. Can the user switch models in the web application?
    - Yes, the prediction interface features a model selector dropdown allowing predictions via the top-performing model or any of the 4 individual classifiers.

28. What is the purpose of the `/models` route?
    - It displays an interactive benchmark dashboard comparing accuracy, precision, recall, F1-scores, and confusion matrices for all 4 models.

29. Is this system suitable for clinical medical diagnosis?
    - No, it is strictly an educational decision-support screening prototype. Final diagnoses must always be made by qualified healthcare professionals.

30. What are promising future enhancements?
    - Hyperparameter tuning via GridSearchCV, Explainable AI (SHAP feature importance), user authentication, and clinical validation on multi-center hospital datasets.

