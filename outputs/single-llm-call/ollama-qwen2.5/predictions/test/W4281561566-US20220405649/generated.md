# DESCRIPTION

## BACKGROUND

Decision support systems are widely used across multiple industries for tasks such as credit risk assessment, fraud detection, and hospital readmission prediction. These systems rely on heuristic approaches using models trained on historical data to classify inputs into specific categories. The effectiveness of these models depends on their ability to accurately predict outcomes and generalize from the training data to new, unseen data. Traditional machine learning models often achieve higher performance through ensembles of simpler learners, such as decision trees or support vector machines (SVMs). However, these classical models have limitations in handling complex non-linear interactions between features, which can be addressed by quantum machine learning (QML) models. QML leverages the principles of quantum mechanics to enhance model performance, but it also faces challenges such as limited practical demonstrations and the complexity of model selection and training processes.

## SUMMARY

The present invention introduces an ensemble generation method for quantum support vector machines (QSVMs) to enhance decision support systems. The system embodiment includes a computer-implemented method that automates the model selection and training process, enabling users without a physics background to utilize advanced QML models. This system offers several advantages over conventional QML models, including improved performance on complex datasets and broader exploration of feature spaces. The computer-implemented method involves a boosting procedure that iteratively selects and combines top-performing quantum kernel-based learners, ensuring stable and accurate predictions. Additionally, the invention includes a computer program product that facilitates the implementation of the ensemble generation method on various computing platforms, including cloud environments. The modified plurality of quantum kernels provides enhanced flexibility and adaptability, leading to better model performance.

## DETAILED DESCRIPTION

The present patent application describes an innovative approach to enhancing decision support systems using quantum machine learning (QML) techniques. Specifically, it focuses on the development and implementation of an ensemble method for Quantum Support Vector Machines (QSVMs). The invention automates the model selection and training process, making QML more accessible and user-friendly.

The system is designed to run on universal quantum computing platforms with gate-based architectures, such as those implemented in IBM Quantum System One. It utilizes shallow circuits for kernel functions, which are stronger than typical weak classifiers discussed in the literature. The ensemble method modifies traditional boosting techniques by performing a grid search for the best model at each iteration and enforcing exploration of different model architectures through parameter constraints.

The computer-implemented method involves several key steps:
1. **Data Preparation**: The input data is split into training, validation, and testing datasets. This ensures that the model can be trained, tuned, and evaluated effectively.
2. **Feature Mapping**: A set of feature maps is defined for grid search, allowing the exploration of different Hilbert spaces and decision boundaries. For example, a feature map on n-qubits can be defined using Hadamard gates and Pauli rotations.
3. **Model Selection**: The algorithm performs a grid search to select the best model based on validation dataset performance. Parameters such as Pauli rotation factors and regularization parameters are varied to find optimal configurations.
4. **Boosting Procedure**: The boosting method iteratively selects and combines top-performing quantum kernel-based learners. Weights are updated for misclassified examples, and early stopping conditions are checked to prevent overfitting.
5. **Ensemble Construction**: Once the boosting procedure is complete, the final model object is returned, which can be used to make predictions on new data.

The computer program product includes software components that facilitate the implementation of the ensemble generation method on various computing platforms. It supports cloud environments and provides user-friendly interfaces for model training and evaluation. The modified plurality of quantum kernels allows for broader exploration of feature spaces, leading to improved model performance.

Numerical simulations have been conducted to validate the effectiveness of the proposed method. Experiments were performed on simulated datasets such as moons, circles, and XOR, with 50 statistically independent datasets generated for each type. The results show that the boosted QSVM ensemble outperforms single QSVMs in many cases, achieving comparable or better performance than classical models like SVM and XGBoost.

In conclusion, the present invention provides a robust and flexible solution for enhancing decision support systems using quantum machine learning techniques. By automating the model selection and training process, it makes advanced QML models more accessible to users without a physics background. The ensemble method ensures stable and accurate predictions, making it a valuable tool for data scientists across multiple industries.

## Data and Data Encoding

In this work, we consider classical examples of generated data, such as moons, circles, and XOR datasets. These datasets allow us to create many different scenarios to accumulate statistics on model performance. The data is split into training, validation, and testing datasets. The validation dataset is used for hyperparameter tuning during the grid search process, while the testing dataset is completely hidden from the training phase to ensure unbiased evaluation.

Following best practices, we define a feature map on n-qubits using Hadamard gates and Pauli rotations. For data with two features, a set of feature maps can be utilized for grid search, as shown in Fig. 1. This allows us to explore different model architectures and feature spaces systematically.

## Ensemble Structure

The traditional AdaBoost variant of boosting relies on weak learners such as decision stumps. However, in this work, we consider support vector machines on quantum kernels (QSVMs), which are not weak learners. Therefore, we modify the boosting method to suit QSVMs. The algorithm receives training and validation datasets along with grid search parameters, including Pauli feature map sets, Pauli rotation factors, and regularization parameters.

The algorithm performs a grid search using the validation dataset to select the best model at each iteration. Early stopping conditions are checked to prevent overfitting. These conditions include perfect classification on the training dataset, performance worse than random guessing, and reaching the maximum number of classifiers. The selected feature map is excluded from subsequent iterations to force exploration of different Hilbert spaces.

Once any stopping condition is satisfied, the final model object is returned, which can be used to make predictions on new data by weighted majority vote of classifiers included in the ensemble.

## Numerical Simulation Results

Numerical simulations were conducted to validate the effectiveness of the proposed method. Experiments were performed on simulated datasets such as moons, circles, and XOR, with 50 statistically independent datasets generated for each type. Each dataset was split into training, validation, and testing subsets equally.

The boosted QSVM ensemble was trained for each dataset, and performance metrics were compared against classical models like SVM and XGBoost. The parameter grids for SVM included RBF and linear kernels, regularization parameters ranging from 0.1 to 100, and gamma parameters for the RBF kernel ranging from 0.0001 to 10. For xgboost, the parameter grid was constructed following established practices.

The results showed that the boosted QSVM ensemble performed comparably or better than classical models in many cases. On the XOR dataset, performance was comparable across all models. The boosted QSVM struggled on the moons dataset but excelled on the circles dataset with median accuracy of 100%. The ensemble size varied depending on the difficulty of the dataset, with more difficult datasets requiring larger ensembles.

## Conclusion

Data scientists across multiple industries continue to seek the best-in-class machine learning models to gain a competitive edge. Quantum machine learning (QML) holds the promise of achieving even higher performance than classical models due to enhanced feature spaces. The approach discussed in this work adapts and extends traditional ensemble building practices to QML, pushing the limits of model performance.

The numerical results demonstrate that boosted QSVM ensembles outperform single QSVMs and can match or exceed the accuracy of non-quantum models. This innovation makes advanced QML models more accessible and user-friendly, enabling broader adoption in various applications. The automated model selection and training process ensures stable and accurate predictions, making it a valuable tool for data scientists.