
Prerequisites
--------------
Get access to World of Code servers.

**Note:** the code is written to be language-agnostic. 
So far we trained models for Python only. 
To reproduce the results, replace `<lang>` in the commands with `PY` 

**Note 2:** WoC is versioned, with only 2..3 last versions available at any 
time. In this study, we used version `Q`, but at the moment of reproduction
it might not be available. Due to this fact and some randomness involved in
sampling, the intermediate data files will not be exactly the same as the ones
shared with this package. We expect, however, the final results to be very
similar regardless of the version used.

Preprocessing
---------------
On da4 (one of WoC servers), run:
`./preprocess.py PY -v`

As a result, you should get these files ():

- `<lang>_project_imports.csv` - 
    two columns (WoC project id + comma-separated imports)
- `<lang>_dev_imports.csv` - source of training data
    two columns (email + comma-separated imports), low usage imports omitted
- `<lang>_dev_commit_counts.csv` - will be used to filter out bots
    two columns (email + count)
- `<lang>_namespace_counts_by_project.csv` - will not be used
    two columns (namespace + count)
- `<lang>_namespace_counts_by_dev.csv` - filter out infrequent namespaces
    two columns (namespace + count)
- `<lang>_vocab.csv`
    one column, low usage ones are already filtered out

All files have no header.

Move these files to directory `data` (create if necessary)



Get package namespaces
---------------------

Make sure folder `ns_whitespaces` exists. If it contains a set of non-empty csv 
files `<lang>_whitelist.csv`, it is safe to skip this step.

**Note:** This step takes few days on a pretty powerful machine. 
Consider using precomputed files.

- download latest libraries.io archive (1.4.0 was used for this study)
- extract and put projects_with_repository_fields*.csv file in this directory
- run:
```bash
./get_ns_whitelist.py projects_with_repository_fields-1.4.0-2018-12-22.csv \
    --save-path=ns_whitelist
```

As a result, you'll get a bunch of `<lang>_whitelist.csv` files in 
`ns_whitelist` directory.



Train models
--------------

Run:

`python train_model.py <lang> <embedding_size> --mode=<mode> `,

Embedding size is self explanatory. Mode is one of `{dev, proj}`, to train on
developer or project profiles, respectively.

As a result of this step, you'll get a trained model saved in `models` directory.



Evaluation
-------------

To evaluate a model, run:

`python validate.py <model_path>`

The output of this file will contain results of analogical reasoning,
nearest neighbor-based prediction, and competing projects benchmarks.



