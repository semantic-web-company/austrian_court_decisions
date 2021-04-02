# NER for Austrian court decisions

Code to repeat the results of Annotating Entities with Fine-Grained Types in Austrian Court Decisions.

## Variables and config

All the variables are stored in `config.toml` and are read during the execution of the code. Please, check this file and modify the entries.
The datasets with original general purpose NER annotations (`nif_annotations_folder`), with manual annotations (`manual_annotations_folder`), and with manual verified samples (`manual_verified_folder`) are available at `https://doi.org/10.5281/zenodo.4625767`. Please, download them and then paste the correct pathes to those dataset into the `config.toml`.

## How to run

1. Create a virtual environment with Python3.7

```bash
pip3.7 install virtualenv
python3.7 -m virtualenv MyEnv
source MyEnv/bin/activate
```

2. Install dependencies

```bash
pip3.7 install -r requirements.txt
```

3. Pre-train the model on `wic-tsv-de`

```bash
python3.7 HyperBertCLS.py
```

4. Repeat results of experiments

```bash
python3.7 evalute_on_manually_verified.py 
```

5. Produce final results 

```bash
python3.7 cybly_type_verification.py 
```