Here is the outline of the desired patent application.
Per bullet point, write roughly 800 words.

Example outline (bullet points are the lines starting with '- '):
## DESCRIPTION OF THE INVENTION
- describe discovery of ODAM protein in human epithelial cancers
- describe method for aiding in diagnosis and management of cancer
- describe specific embodiments of the invention
- describe methods for determining presence of ODAM or anti-ODAM antibodies

In the example above, each line beginning with '- ' is a bullet point.

```md
# DESCRIPTION

## BACKGROUND

- introduce decision support systems
- motivate quantum machine learning models
- limitations of quantum machine learning models

## SUMMARY

- introduce ensemble generation
- describe system embodiment
- advantage of system embodiment
- describe computer-implemented method
- advantage of computer-implemented method
- describe computer program product
- advantage of computer program product
- describe training and performance metric
- advantage of modified plurality of quantum kernels

## DETAILED DESCRIPTION

- introduce patent application structure
- define scope of embodiments
- describe problems with other feature space discovery implementations
- introduce boosting procedure to generate feature space
- define boosting procedure
- describe advantages of boosting procedure
- introduce ensemble of quantum kernel-based learners
- describe application of ensemble to machine learning tasks
- introduce computer processing systems, methods, apparatus, and program products
- describe technical improvements over conventional quantum machine learning models
- illustrate block diagram of system 100
- describe server 102 components
- introduce ensemble component 110
- describe communications component 112
- introduce memory 116
- describe system bus 118
- introduce processor 120
- describe one or more networks 104
- introduce input devices 106
- describe quantum computers 108
- introduce quantum data plane
- describe control processor plane
- introduce control and measurement plane
- describe qubit technologies
- introduce trapped ion qubits
- describe superconducting qubits
- introduce photonic qubits, quantum dot qubits, gate-based neutral atom qubits, semiconductor qubits
- describe control processor plane functions
- introduce control and measurement plane functions
- describe quantum kernel-based learners 122
- introduce quantum algorithms for kernel-based machine learning
- describe QSVM models and quantum NN models
- introduce machine learning tasks
- describe supervised and unsupervised learning protocols
- introduce ensemble component 110 functions
- describe one or more boosting procedures
- introduce selection of top performing quantum kernel
- describe weighting and combining quantum kernel-based learners
- introduce modification of quantum kernel search
- describe control of quantum kernel-based learners
- introduce input data processing
- describe feature mapping and quantum kernel calculation
- introduce input data entry
- describe input data processing for machine learning tasks
- introduce splitting of input data into training and testing datasets
- describe processing of input data for machine learning tasks
- introduce definition of boosting procedure parameters
- describe storage of training datasets, testing datasets, and boosting procedure parameters
- illustrate diagram of system 100 with training component 202 and testing component 204
- describe training component 202 functions
- introduce testing component 204 functions
- describe training of quantum kernel-based learners
- introduce calculation of kernel matrix
- describe permutations of feature maps and Pauli rotation factors
- introduce boosting procedure
- test performance of quantum kernel-based learners
- generate performance metric
- select quantum kernel and associate learner
- illustrate feature maps
- describe qubit connections and quantum gates
- modify quantum kernel search
- analyze performance metrics
- select top performing quantum kernel-based learner
- exclude selected quantum kernel
- adjust weight values of samples
- assign weight value to learner
- combine top performing learners
- illustrate weight value determinations
- train learners on samples
- test learners on training dataset
- generate performance metrics
- adjust sample weights
- assign weight value to learner
- exclude selected learner
- repeat iterations
- terminate boosting procedure
- combine top performing learners
- assign weight value to learner
- illustrate algorithm
- fit classifiers
- choose best classifier
- compute errors
- compute learner weights
- adjust sample weights
- sample training dataset
- combine learners
- implement quantum machine learning model
- generate reports
- detect overfitting
- combine with other models
- receive input data and feature maps
- train quantum kernel-based learners
- generate performance metrics
- select quantum kernel-based learner
- repeat iterations
- terminate boosting procedure
- combine top performing learners
- assign weight value to learner
- implement quantum machine learning model
- generate reports
- detect overfitting
- combine with other models
- describe system components
- describe training component
- describe testing component
- describe modification component
- describe error weight component
- describe learner weight component
- describe ensemble component
- describe computer-implemented method 900
- modify quantum kernels
- adjust learner weight
- adjust error weight
- determine termination event
- perform boosting procedure
- generate ensemble of quantum kernel-based learners
- employ ensemble in quantum machine learning models
- introduce cloud computing
- define cloud computing characteristics
- describe on-demand self-service
- describe broad network access
- describe resource pooling
- describe rapid elasticity
- describe measured service
- introduce service models
- describe software as a service
- describe platform as a service
- describe infrastructure as a service
- introduce deployment models
- describe private cloud
- describe community cloud
- describe public cloud
- describe hybrid cloud
- describe cloud computing environment
- illustrate cloud computing environment
- describe functional abstraction layers
- describe hardware and software layer
- describe workloads layer
- describe computer readable program instructions
- describe network for downloading instructions
- describe types of instructions
- describe execution of instructions
- describe flowchart and block diagrams
- describe blocks as modules or segments
- describe alternative implementations
- describe computing environment
- describe program modules
- describe distributed computing environments
- describe memory and memory units
- describe computer-readable storage media
- describe machine-readable storage media
- describe communications media
- describe tangible and non-transitory media
- describe accessing storage media
- describe modulated data signal
- describe example environment
- describe computer components
- describe processing unit
- describe system memory
- describe system bus
- describe ROM and RAM
- describe BIOS
- describe internal hard disk drive
- describe external storage devices
- describe optical disk drive
- describe interfaces for storage devices
- describe drives and storage media
- describe program modules and data
- describe operating system
- describe application programs
- describe emulation technologies
- describe hypervisor
- describe virtual machine
- describe runtime environments
- describe containers
- describe security module
- describe input devices
- describe display device
- describe networked environment
- describe remote computer
- describe cloud storage systems
```

You need to draft a complete patent application that strictly follows the outline's section order and headings. Do not skip any bullet points. Use formal patent language. The generated patent must not be shorter than the research paper in word count.

Here is the research paper that describes the invention:

```md
# I. INTRODUCTION

D ECISION support systems across multiple industries rely on heuristic approaches using models trained on historical data. For example, credit risk, propensity, attrition, fraud, and hospital readmission risk models classify data into two classes given a set of input features. The goal is to train analytical models that give the most accurate predictions, retain stable performance over time and utilize fewer features possible by efficiently extracting information from available data. Generally, the better the model can describe complex non-linear interactions between features, the higher its performance and ability to generalize the data will be. Conventional, non-quantum, machine learning models with higher performance are often achieved through ensembles of simpler learners. See, for example [1], [2], [3].

Havlicek, et al. [4] implemented a quantum support vector machine classifier (QSVM) on a superconducting processor. Originally proposed in [5], QSVM exploits a high-dimensional quantum Hilbert space to obtain an enhanced solution. This enhancement can be achieved through controlled entanglement and interference, which is inaccessible for classical support vector machines. However, superior performance of QSVM or other quantum approaches compared to traditional machine learning models is yet to be demonstrated on a practical dataset. Park et al. [6] demonstrated improvements to QSVM compared to classical SVM by using parameterized shallow unitary transformations for feature maps with rotation and regularization. Wu et al. [7] provided benchmarks comparing performance of QSVM built using a simulator and physical hardware with classical SVM and xgboost. Those benchmarks built using three different platforms, IBM Quantum, Google Tensorflow Quantum and Amazon Bracket, indicated similar performance of QSVM compared to its classical counterparts on a practical dataset. Another recent paper by Glick et al. [8] discusses a class of covariant kernels and quantum advantage for problems where the data satisfies a group structure.

The idea of boosting quantum machine learning models was previously discussed by Neven et al. [9] in the context of adiabatic quantum computing implemented on D-Wave annealers, where authors used one level decision trees as weak classifiers. Papers by Schuld et al. and Abbas et al. [10], [11] discussed quantum ensembles of quantum classifiers primarily from a perspective of speedup due to parallel calculation.

In general, quantum machine learning (QML) models consist of data encoding into qubits, a variational quantum circuit with trainable parameters, a classical cost function and an optimization algorithm. Most of the QML models constructed this way are mathematically related to quantum kernel methods [12]. Notably, initial state preparation and subsequent unitary transformation with input features is carried out through a circuit called feature map. Unlike other types of machine learning algorithms, the choice of initial feature map in quantum support vector machines (QSVM) could yield unique decision boundaries, making QSVMs with different feature maps independent from each other. This characteristic of QSVM is well suited for implementing boosting algorithms, however, given a large number of choices of feature maps the automation of feature map/model selection and training process is quite desirable.

In addition to achieving better performance with quantum models, there is a need to make them more user-friendly by automating the model selection and training process. Currently, model architectures are often derived from well-known physical models, e.g. an Ising model and respective Hamil-tonian were used for feature mapping. Thus, an automated procedure can augment some of the unnecessary complexity of the existing models for users without a physics background as well as assist in discovering new model architectures.

The approach presented in the current work is different from results discussed in the referenced sources in the following respects. First, we are focused on universal quantum computing with gates that can run on superconducting qubits as implemented, e.g. in IBM Quantum System One. Secondly, even though we consider shallow circuits for kernel functions, those tend to be stronger than typical weak classifiers discussed in the literature, which is mostly based on decision trees. Thirdly, we implement an automated model selection on every boosting step to choose from different topologies and thus explore wider feature and model spaces. This process can be used to search for alternatives for broadly used Ising-type models. Fourthly, our approach is not constrained to classification tasks, it can be equally applied to regression tasks. Lastly, the focus of many prior results was on the possibility of quantum speedup for classical procedures, whereas our focus is primarily on the development of models with higher performance. In this work we are applying our approach to the problem of binary classification and use classification accuracy on the test sample as a measure of performance.

## Our main contributions are:

• A new ensemble method for QSVM that enhances model performance, when the data is difficult to model for a single learner. • Hyperparameter optimization for QSVM.

• Simulation on multiple datasets to ensure stability of results.

# A. Data and Data Encoding

In this work we consider classical examples of generated data, such as moons, circles and XOR. This allows creation of many different datasets to accumulate statistics of model performance.

Following the best practice, the data is split into training, validation and testing datasets. A validation dataset is used for hyperparameter tuning in the process of grid search for the best model on every step of the boosting procedure. A testing dataset is completely hidden from the training and is used to compare different models.

Following [4] we define a feature map on n-qubits as

where

Here H is the Hadamard gate and P i ∈ {I, X, Y, Z}. A set of feature maps that we can utilize for a grid search, when the data has two features is shown in Fig. 1.

# B. Ensemble Structure

The traditional AdaBoost variant of boosting relies on weak learners, such as decision stumps, that are trained on every iteration [3]. For each subsequent iteration it emphasizes examples that were previously misclassified by calculating and assigning or updating their weights. The final prediction is calculated by weighted majority vote of classifiers. In this work we consider support vector machines on quantum kernels k( x i , x j ) = | U Φ ( x i )|U Φ ( x j )| 2 that we call Quantum Support Vector Machines (QSVM). QSVM is not a weak learner, so we modify the boosting method as shown in Algorithm 1.

In the beginning the algorithm receives training and validation datasets as well as grid search parameters. In this work we consider the following parameters: Pauli feature map set as shown in Fig. 1, the Pauli rotation factor, which is a multiplier to the Pauli rotations (alpha) and a regularization parameter (C) for sklearn's support vector classifier (SVC). We vary alpha in the interval (0; 2], C in [1; 100]. All examples are initially assigned a weight of 1. Grid search uses a validation dataset to select the best model. After the best model is selected we check early stopping conditions:

1) Estimator is perfect, i.e. estimator error on the training dataset is ≤ 0. 2) Estimator is as bad as random guessing or worse, i.e.

estimator error is ≥ 0.5 for binary classification or ≥ 1 -

## 1

N classes for multiclass.

3) The maximum number of classifiers is reached. The feature map selected on the current iteration is excluded from the grid search for next iterations. This is important to force the model to explore a broader Hilbert space and, consequently, different decision boundaries by choosing other feature maps for the quantum kernel. Finally, the weights are updated as shown in Algorithm 1. Once any stopping condition is satisfied then the final model object is returned. This object can be used to build predictions for new samples as a weighted majority vote of classifiers included in the model.

It is worthwhile to highlight differences of the approach presented here from more traditional boosting: 1) we perform a grid search for the best model on each iteration of the algorithm, 2) we enforce exploration of different model architectures through parameter grid constrains.

Identifying the optimal number of estimators and ensemble pruning is generally outside of the boosting method description and is up to the user. In this work we will choose the optimal number of estimators based on the minimum error on the validation sample.

# C. Numerical Simulation Results

First, we run experiments on simulated data created with functions available in scikit-learn (see Fig. 2). This allows us to create a number of statistically independent datasets and obtain averaged performance metrics. In this study we chose to generate 50 datasets of each kind: XOR, moons and circles.

Each dataset has 150 observations for training, validation and testing split equally between those subsets. We train a boosted QSVM as described above for each dataset. For comparison, we also train an SVM and xgboost. The parameter grid Check early stopping conditions for perfect and worse than random guessing classification.

## 5:

Exclude selected feature map from grid parameters for next iterations. Compute α m = log((1 -err m )/err m ) (estimator weight)

for the SVM includes RBF and linear kernels, regularization C ranging from 0.1 to 100, gamma parameter for RBF kernel ranging from 0.0001 to 10. The parameter grid for xgboost was constructed following [13].

The results are shown in Fig. 3. The performance on the XOR dataset seems comparable across the three models. Boosted QSVM struggles to achieve comparable performance on the moons dataset, but works best on the circles dataset with median at 100% accuracy.

An interesting question is whether a Boosted QSVM actually benefited from the ensemble and if so then how much improvement did it provide. It turns out that only about 31% of Boosted QSVM models contain more than 1 estimator in the ensemble. Table I shows mean and maximum ensemble size by dataset. The more difficult the dataset for QSVM is, the larger the ensemble seems to be: more than 1 estimator is barely used for circles data, while 3.8 estimators on average are used for moons data. We have also investigated whether there is a performance gain from having multiple classifiers. Table II shows classi-  fication accuracy improvement from an ensemble of QSVM classifiers compared to a single QSVM. There is a small sample size for circles data, where even a single QSVM is doing well. There is an average of 4.2% and 7.5% classification accuracy improvement for XOR and moons.

## III. CONCLUSIONS

Data scientists across multiple industries continue to push limits in their search for the best-in-class machine learning model that would provide a competitive edge. Quantum ma- chine learning holds a promise of even higher performance than classical due to enhanced feature spaces. The approach discussed here is derived and adapted from the best ensemble building practices that worked well in traditional machine learning and thus should push the limits of model performance even further. Examples discussed in this work show that boosted QSVM ensembles outperform single QSVMs that in some cases allows them to match accuracy of non-quantum models, and in other cases -even exceed it.
```
