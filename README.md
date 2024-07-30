# Job <-> Talent Matching engine

> Product

This project aims at build a lightweight search & ranking component to match jobs with talents.

## Getting started

To set up and run this project, follow the below instructions:
- Create and activate a virtual environment
```bash
mamba create -n py311-job-talent-matching-engine python=3.11 "poetry>=1.7"
mamba activate py311-job-talent-matching-engine
```
> Note: you have the option to use conda with the same commands,
> or you can install mamba by following the provided instructions :

```bash
brew install miniforge
conda install mamba
```

- Clone and enter repository
```bash
git clone https://github.com/guillaumelf/job-talent-matching-engine
cd job-talent-matching-engine
```

- Install all project's dependencies
```bash
poetry install
```

- Run the project
```bash
python -m src
```

The main code is located in the `src` folder, data exploration & model training notebooks can be found in the `notebooks` directory.