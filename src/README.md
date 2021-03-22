# Homography Ranking

This project contains the experimental part of the **homography ranking**
algorithm.

## Running Experiments

In order to run the experiments, the file ``experiment_runner.py`` has to be
used. This script loads a configuration settings from the``config.py`` file.
Subsequently, the specified experiments are executed to obtain a single file
containing the results. This file can then be analyzed in the 
``stats_analysis.ipynb`` notebook.

## Running On Custom Data

Running the script.

```
python rank_homographies.py ../test_data/warped_points_test.npy ../test_data/target_points_test.npy
```
