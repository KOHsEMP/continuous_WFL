# Unified Analysis of Continuous Weak Features Learning with Applications to Learning from Missing Data

This repository contains the scripts for the experiments, as presented in the paper "Unified Analysis of Continuous Weak Features Learning with Applications to Learning from Missing Data", by Kosuke Sugiyama and Masato Uchida (ICML 2025)

## requirements

* Python: 3.9.14

* Library: Please see `requirements.txt`


## Directory Structure

```
codes_continuous/
　├README.md
　├config/
　│　├exp_electricity_base.yaml
　│　├exp_jets_full_base.yaml
　│　├exp_mv_full_base.yaml
　│　└exp_run-or-walk_full_base.yaml
　├exp1_shell/
　│　├exp1_mcar_electricity_full.sh
　│　├exp1_mcar_jets_full.sh
　│　├exp1_mcar_mv_full.sh
　│　└exp1_mcar_run-or-walk_full.sh
　├data/
　│　├electricity/
　│　│　├ ...
　│　├jets/
　│　│　├ ...
　│　├mv/
　│　│　├ ...
　│　└run-or-walk/
　│　 　├ ...
　├libs/
　│　├learning.py
　│　├load_data.py
　│　├models.py
　│　├utils_processing.py
　│　└utils.py
　├requirements.txt
　├exp1.py
　└calc_bound.ipynb
```

* `config`: This is a directory that stores YAML files, which contain arguments that are common and fixed across the experiment scripts.
* `exp1_shell`: Shell scripts for executing experimental programs.
* `data`: This is a directory that stores the datasets used in the experiments. When executing the program, specify the path to the data directory as an argument.
* `libs`: This is a directory that stores functions and other utilities used in main.py.
* `exp1.py`: This is the experiment script.
* `calc_bound.ipynb`: The code calculates the derived error bounds. Visualization of experimental results is also provided.


## Datasets download Links

* [hls4ml_lhc_jets_hlf; Jets](https://www.openml.org/search?type=data&status=active&qualities.NumberOfInstances=between_10000_100000&qualities.NumberOfClasses=gte_2&format=ARFF&id=42468&sort=runs)

    * Unzip `hls4ml_HLF.arff` and place `hls4ml_HLF.arff` into `./data/jets/`.

* [electricity; Electorisity](https://www.openml.org/search?type=data&status=active&id=151&sort=runs)

    * Unzip `electricity-normalized.arff` and place `electricity-normalized.arff` into `./data/electricity/`.

* [mv; Mv](https://www.openml.org/search?type=data&status=active&sort=runs&order=desc&qualities.NumberOfInstances=between_10000_100000&id=881)

    * Unzip `mv.arff` and place `mv.arff` into `./data/mv/`.

* [Run_or_walk_information; Run-or-Walk](https://www.openml.org/search?type=data&status=active&id=40922&sort=runs)

    * Unzip `phpMD2hR6.arff`, rename `phpMD2hR6.arff` to `run-or-walk.arff`, and store it in `./data/run-or-walk/`.


## How to Execute Experiments

```bash

# full experiments using Electricity dataset 
bash ./exp1_shell/exp1_mcar_electricity_full.sh

```

The explanation of the main arguments is follow:

**Experiental Settings:**

* `dataset_name`: Using dataset name. Please select from ['bank', 'adult', 'kick', 'census'].

* `data_dir`: The path of `data` directory.

* `output_dir`: Path of output directory for log data. 

* `weak_cols`: List of features to be weak features

* `miss_pattern`: Please select from ['MCAR', 'MAR_logistic', 'MNAR_gsm'].

* `miss_rate`: Missing rate

* `sm_k`: the parameter for miss_pattern=='MNAR_gsm'.

* `sm_prop_latent`: the parameter for miss_pattern=='MNAR_gsm'.

* `rm_k`: the value of $k$.

* `sample_size`: All data size. If sample_size = -1, we use all data

* `test_rate`: Test data rate

* `use_train_size`: Size of the training data. Randomly selected from samples not assigned to test data.

* `seed`: Random Seed

* `pred_arch`: Architecture for a label prediction model. Please choose 'mlp'.

* `hd`: The size of Hidden dimension for arch=='mlp'.

* `lr`: Learning rate

* `bs`: Batch size

* `ep`: The number of epochs

* `wd`: Weight decay

* `pred_loss`: Loss function for learning label prediction model.

* `est_error_mse`: the mean squared error of feature estimation models.
