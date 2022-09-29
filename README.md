# Modelling in Biology - Python Implementation

This repository contains the problem sheets and example code for the Modelling in Biology undergraduate course study sessions.

## Setup

### Conda (recommended)

1. Install Conda by following the [user guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Create the environment by calling
```
$ conda env create -f environment.yml
```
3. Activate the environment
```
$ conda activate mib
```

### Pip

1. Create a virtual environment to run the scripts in
```
python -m venv .
```
2. Activate the venv
```
source bin/activate
```
3. Install the required packages
```
pip install -r requirements.txt
```

## Usage

To generate all results from the set of worksheets, simply run 

```
$ ./run_all.sh
```

Note that you might have to make ```run_all.sh``` executable:

```
chmod +x run_all.sh
```