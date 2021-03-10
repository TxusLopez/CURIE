# CURIE

---------
RESULTS
---------
Detailed results are available in "res.xlsx" file. The first tab corresponds to those experiments referred as F1 in the paper. The second tab corresponds to those experiments referred as F2 in the paper. The last tab is a mean of F1 and F2 experiments, and can be considered as final results, which have been presented in the paper in Table 2.

-------------
DATASETS
-------------
All datasets have been uploaded to Harvard Dataverse repository: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5OWRGB. The scikit-multiflow framework, as one of the most commonly accepted libraries in stream learning, allows generating a wide variety of synthetic data in order to simulate a realistic occurrence of drifts. Here, and in Section 5.1 of the manuscript, the detail of the datasets:

- With "Sine" generator: it is ruled by a sequence of classification functions. "Sine_A" refers to abrupt cases and "SineG" to gradual ones. In the case of "Sine_F1", the order of the functions is SINE1, reversed SINE1, SINE2, and reversed SINE2. For "Sine_F2" the  order is namely, reversed SINE2, SINE2, reversed SINE1, and SINE1. Therefore, "Sine" stream  generator provides 4 different datasets: Sine_A_F1, Sine_A_F2, Sine_G_F1, and Sine_G_F2. They consist of 2 numerical features without noise, and a balanced binary class.
- With "Random Tree" generator: it is ruled by a sequence of tree random state functions, i.e. the seed for the random generation of trees. We have chosen 4 seeds (9856, 2563, 8873, 7896), and we have altered their order to produce 4 different datasets. "RT_A" refers to abrupt cases and "RT_G" to gradual ones; then theyare: RT_A_F1,RT_A_F2,RT_G_F1,  and RT_G_F2. The parameters max_tree_depth, min_leaf_depth, and fraction_leaves_per_level were set to 6, 3, and 0.15 respectively. They consist of 2 numerical features without noise, and balanced binary class.
- With "Mixed" generator: it is ruled by a sequence of classification functions. "Mixed_A" refers to abrupt cases and "Mixed_G" to gradual ones. In the case of "Mixed_F1", the  order of the functions is 0-1-0-1. For "Mixed_F2" the order is reversed 1-0-1-0. Therefore, "Mixed" stream generator provides 4 different datasets: Mixed_A_F1, Mixed_A_F2, Mixed_G_F1, and Mixed_G_F2. They consist of 4 numerical features without noise, and a balanced binary class.
- With "Sea" generator: it is ruled by a sequence of classification functions. "Sea_A" refers to abrupt cases and "Sea_G" to gradual ones. In the case of "Sea_F1", the order of the functions is 0-1-2-3. For "Sea_F2" the order is reversed 3-2-1-0. Therefore, "Sea" stream generator provides 4 different datasets: "Sea_A_F1", "Sea_A_F2", "Sea_G_F1",and "Sea_G_F2". They consist of 3 numerical features, a balanced binary class, andwith the probability that noise will happen in the generation of 0.2 (probability range between 0 and 1).
- With "Stagger" generator: it is ruled by a sequence of classification functions. "Stagger_A" refers to abrupt cases and "Stagger_G" to gradual ones. In the case of "Stagger_F1", the order of the functions is 0-1-2-0. For "Stagger_F2" the order is reversed 2-1-0-2.  Therefore, "Stagger" stream generator provides 4  different datasets: "Stagger_A_F1", "Stagger_A_F2", "Stagger_G_F1", and "Stagger_G_F2". They consist of 3 numerical features without noise, and a balanced binary class.


-------------
EXPERIMENTS
-------------
All the experiments can be performed with the file "curie_def.py". Here, we can select:

- Datasets and types with the variables "datasets" and "tipos" respectively.
- CURIE parameters: bins_margin,mutation_period,num_mutantneighs_fordetection,preparatory_size,sliding_window_size,radius, and n_bins.
- The path to save results in variable "path_saving_results".
- The path to read the datasets in variable "path".
- The functions order in "functions_order" variable to launch the experiments of F1 or F2 as explained in the paper.
- The variable "curie" for the CURIE detector.
- The variable "detectores_ref" for all the detectors.
- The variable "learners_ref" for all the base learners.
- The results are stored in different variables for scores (prequential accuracy), times and rams for calculating RAM-Hours metric, and detections.
- These results are then stored in a "temp.csv"
- Finally, Friedman and Nemenyi tests can be performed with the corresponding results for each metric (prequential accuracy, RAM-Hours, distance to the drift, and MCC).

--------------------
CURIE source code
--------------------
The source code for CURIE can be found in the file "CA_VonNeumann_estimator.py".
