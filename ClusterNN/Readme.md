"Data_import.py" and "tools.py" are used to import data.

The model is established in the "ClusterNN.py";

"ClusterNN_without_linear_layer.py" is a model where the additional linear layer has been neglected.

"_seasonal_occurrence_prob.ipynb" displays the seasonal characteristics obtained from the 21-year pilot reports and the hazard factor, respectively. The correlation coefficient is obtained.

The "model_selection" folder contains the codes correpond to the model selection, such as the number of the hazardous labels in the training dataset, the additional linear layer, and so on.

A comparative analysis is conducted in the "Comparison" folder, where the detection performance of this model, CatB, IForest, and XGOB are compared.

"main_transfer_learning.py" intends to generalize the model to the northern runway by transfer learning.

"main_tsne_train_cmp.py" intends to train the model and visualize the iterative refinements of the cluster feature space.
