# AbDPP: Target-oriented Antibody Design with Pre-training and Prior Biological Structure Knowledge

## Cite

To be added...

## Envs

The code requires python 3.7+ and packages listed in the requiements.txt. To create environment to run the code, you can
simply run:
> pip3 install -r requirements.txt

## Database

The training data was downloaded from several public database shown below, and a detailed description of the
preprocessing steps can be found in the paper.

The main datasets include:
[OAS Dataset](https://opig.stats.ox.ac.uk/webapps/oas/),
[SARS-CoV-2 Dataset](https://opig.stats.ox.ac.uk/webapps/covabdab/),
[RAbD Dataset](http://dunbrack2.fccc.edu/%20PyIgClassify).

## Usage

AntiTranslate.py includes our generation model, AntDPP.

The primary inference step for generation is incorporated in staircase_generation.py.

Running it with the command:
> `python Staircase_generation.py`.

When generating CDR region, you can prepare data samples by yourself, we also provide examples here in the data file.
