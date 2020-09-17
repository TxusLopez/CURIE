# CURIE

---------
RESULTS
---------
Detailed results are available in "res.xlsx" file. The first tab corresponds to those experiments referred as F1 in the paper. The second tab corresponds to those experiments referred as F2 in the paper. The last tab is a mean of F1 and F2 experiments, and can be considered as final results, which have been presented in the paper in Table 2.

-------------
DATASETS
-------------
All datasets have been uploaded to Harvard Dataverse repository: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5OWRGB. The details of these datasets can be found in the paper in Section 5.1.

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
