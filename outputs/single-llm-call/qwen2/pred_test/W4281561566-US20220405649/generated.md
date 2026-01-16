# DESCRIPTION

## BACKGROUND

In the realm of decision support systems, various industries rely heavily on heuristic approaches that leverage models trained on historical data. These models are crucial for tasks such as credit risk assessment, customer propensity analysis, attrition prediction, fraud detection, and hospital readmission risk evaluation. The primary objective is to develop analytical models that offer the most accurate predictions, maintain stable performance over time, and utilize the fewest possible features by efficiently extracting information from available data. The ability of a model to describe complex non-linear interactions between features is directly correlated with its performance and generalization capabilities. 

Conventional, non-quantum machine learning models often achieve higher performance through ensembles of simpler learners, such as decision trees. However, the advent of quantum computing has introduced new possibilities for enhancing these models. For instance, Havlicek et al. implemented a quantum support vector machine classifier (QSVM) on a superconducting processor, exploiting a high-dimensional quantum Hilbert space to obtain enhanced solutions through controlled entanglement and interference. Despite these advancements, demonstrating superior performance of QSVM or other quantum approaches compared to traditional machine learning models on practical datasets remains a challenge.

Recent studies have explored various aspects of quantum machine learning (QML). Park et al. demonstrated improvements to QSVM by using parameterized shallow unitary transformations for feature maps with rotation and regularization. Wu et al. provided benchmarks comparing the performance of QSVM built using simulators and physical hardware with classical SVM and xgboost, indicating similar performance on practical datasets. Glick et al. discussed a class of covariant kernels and quantum advantage for problems with group structures. Neven et al. explored boosting quantum machine learning models in the context of adiabatic quantum computing, while Schuld et al. and Abbas et al. discussed quantum ensembles of quantum classifiers from a speedup perspective.

Despite these advancements, there is a need to make quantum models more user-friendly by automating the model selection and training process. Current model architectures often derive from well-known physical models, such as the Ising model, which can introduce unnecessary complexity for users without a physics background. An automated procedure can help discover new model architectures and simplify the use of quantum models.

The present invention addresses these challenges by introducing a new ensemble method for QSVM that enhances model performance, especially when the data is difficult to model for a single learner. It also includes hyperparameter optimization for QSVM and simulation on multiple datasets to ensure the stability of results. The approach is not limited to classification tasks and can be equally applied to regression tasks. The focus is on developing models with higher performance rather than on quantum speedup for classical procedures.

## SUMMARY

The present invention provides a novel ensemble method for Quantum Support Vector Machines (QSVM) that significantly enhances model performance, particularly when dealing with complex datasets that are challenging for a single learner. The method involves a modified boosting algorithm that incorporates hyperparameter optimization and automated model selection. Key features of the invention include:

1. **Ensemble Method for QSVM**: The invention introduces a new ensemble method for QSVM that leverages the strengths of multiple quantum models to improve overall performance. Unlike traditional boosting methods that use weak learners like decision stumps, this method employs QSVMs, which are not inherently weak learners. The ensemble is built iteratively, with each iteration selecting the best model from a grid search of hyperparameters and feature maps.

2. **Hyperparameter Optimization**: The invention includes a comprehensive hyperparameter optimization process for QSVM. This process involves varying parameters such as the Pauli feature map set, the Pauli rotation factor (alpha), and the regularization parameter (C) for the support vector classifier (SVC). The optimization is performed using a validation dataset to select the best model at each boosting iteration.

3. **Automated Model Selection**: The invention automates the model selection process by excluding the feature map selected on the current iteration from the grid search for subsequent iterations. This ensures that the model explores a broader Hilbert space and different decision boundaries, leading to more diverse and robust ensembles.

4. **Stability and Performance Validation**: The invention includes numerical simulations on multiple datasets to validate the stability and performance of the boosted QSVM. The simulations are conducted on classical examples of generated data, such as moons, circles, and XOR, to accumulate statistics of model performance. The results demonstrate that the boosted QSVM can achieve higher accuracy compared to single QSVMs and, in some cases, outperform classical models like SVM and xgboost.

5. **Versatility**: The invention is not limited to classification tasks and can be equally applied to regression tasks. This versatility makes the method applicable to a wide range of decision support systems across various industries.

The primary objective of the invention is to provide a robust and efficient method for enhancing the performance of QSVMs, making them more accessible and user-friendly for data scientists and researchers. By automating the model selection and training process, the invention simplifies the use of quantum models and helps discover new model architectures.

## DETAILED DESCRIPTION

### Data and Data Encoding

The invention utilizes classical examples of generated data, such as moons, circles, and XOR, to create multiple datasets for training, validation, and testing. These datasets allow for the accumulation of statistics on model performance. The data is split into training, validation, and testing datasets, with the validation dataset used for hyperparameter tuning and the testing dataset reserved for final model evaluation.

A feature map on \( n \)-qubits is defined as:
\[
U_{\Phi}(x) = H^{\otimes n} \prod_{i=1}^{n} P_i(\theta_i)
\]
where \( H \) is the Hadamard gate and \( P_i \in \{I, X, Y, Z\} \). A set of feature maps is utilized for grid search, allowing the model to explore different configurations of Pauli gates and rotations.

### Ensemble Structure

The ensemble method for QSVM is based on a modified boosting algorithm. Unlike traditional AdaBoost, which relies on weak learners like decision stumps, this method uses QSVMs, which are not inherently weak learners. The algorithm proceeds as follows:

1. **Initialization**: The algorithm receives the training and validation datasets along with grid search parameters. The parameters include the Pauli feature map set, the Pauli rotation factor (alpha), and the regularization parameter (C) for the SVC. All examples are initially assigned a weight of 1.

2. **Grid Search**: On each iteration, a grid search is performed using the validation dataset to select the best model. The grid search varies the Pauli feature map set, the Pauli rotation factor (alpha) in the interval (0; 2], and the regularization parameter (C) in the range [1; 100].

3. **Model Selection**: The best model is selected based on the performance on the validation dataset. Early stopping conditions are checked:
   - If the estimator error on the training dataset is ≤ 0, the estimator is considered perfect.
   - If the estimator error is ≥ 0.5 for binary classification or ≥ 1 - \( \frac{1}{N} \) for multiclass, the estimator is considered as bad as random guessing or worse.
   - If the maximum number of classifiers is reached.

4. **Feature Map Exclusion**: The feature map selected on the current iteration is excluded from the grid search for subsequent iterations to ensure exploration of different model architectures.

5. **Weight Update**: The weights of the examples are updated based on the performance of the selected model. The weight update formula is:
   \[
   \alpha_m = \log\left(\frac{1 - \text{err}_m}{\text{err}_m}\right)
   \]
   where \( \text{err}_m \) is the error of the \( m \)-th classifier.

6. **Final Model**: Once any stopping condition is satisfied, the final model object is returned. This object can be used to build predictions for new samples as a weighted majority vote of the classifiers included in the model.

### Numerical Simulation Results

Numerical simulations were conducted to validate the performance and stability of the boosted QSVM. The simulations were performed on 50 datasets of each type: XOR, moons, and circles. Each dataset had 150 observations split equally between training, validation, and testing subsets. The boosted QSVM was trained for each dataset, and the results were compared with classical models like SVM and xgboost.

The parameter grids for the SVM and xgboost were constructed as follows:
- **SVM**: RBF and linear kernels, regularization C ranging from 0.1 to 100, gamma parameter for RBF kernel ranging from 0.0001 to 10.
- **xgboost**: Parameters were chosen following established guidelines.

The results showed that the boosted QSVM achieved comparable performance on the XOR dataset, struggled on the moons dataset, but performed best on the circles dataset with a median accuracy of 100%. The ensemble size varied depending on the difficulty of the dataset, with more difficult datasets requiring larger ensembles.

The performance gain from the ensemble was also investigated. On average, the boosted QSVM showed a 4.2% and 7.5% improvement in classification accuracy for XOR and moons datasets, respectively, compared to a single QSVM.

### Conclusion

The invention presents a novel ensemble method for QSVM that significantly enhances model performance, especially for complex datasets. The method combines hyperparameter optimization and automated model selection to build robust and diverse ensembles. Numerical simulations demonstrate that the boosted QSVM can achieve higher accuracy compared to single QSVMs and, in some cases, outperform classical models. The versatility of the method, applicable to both classification and regression tasks, makes it a valuable tool for data scientists and researchers across various industries.