# zk-models
This repository aims to provide pipelines for trainig neural networks and proving them by transforming networks to zero-knowledge circuits.
Current implementation supports transforming linear regression model and multi-layer perceptron (MLP) to zk circuit.
You can choose to convert neural network to zk circuit by using Circom or EZKL.

## Training neural network.
*train_linear_regression.py* and *train_mlp.py* provides pipeline for training heart failure prediction model.
You can train your model by running:
```
python train_linear_regression.py
```
or
```
python train_mlp.py
```
The dataset is contained in the repository under name *heart.csv*. 
Specific details about the dataset can be found [here](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

After the training has finished, you can find your model checkpoint under *data* directory.

## Converting neural network using Circom
For circom proof generation, we use Groth16 backend. Therefore, you first need to go through Groth16 specific setups.
They are: 

```
cd circom_data
circom ../circom_circuits/mlp.circom --r1cs --wasm --sym
snarkjs powersoftau new bn128 19 pot19_0000.ptau -v
snarkjs powersoftau contribute pot19_0000.ptau pot19_0001.ptau --name="First contribution" -v
snarkjs powersoftau prepare phase2 pot19_0001.ptau pot19_final.ptau -v
snarkjs groth16 setup mlp.r1cs pot19_final.ptau proof0.key
snarkjs zkey contribute proof0.key proof01.key --name="your name" -v
snarkjs zkey export verificationkey proof01.key verification_key.json
```
after thant, you can run 
```
python prove_model_circom.py
```
for proving your model. We note that current code supports proof generation for MLP. For linear regression, you may want to change file names and proxy calculation for checking performance degradation when neural network gets transformed into zk circuit. We are planning to modify the code so that user can switch between models with more flexibility. 

## Converting neural network using EZKL
For EZKL proof generation, first install EZKL via:
```
pip install ezkl
```
After that, you can generate proof by running:
```
cd ..
python prove_model_ezkl.py
```

## Notes
*prove_model_circom.py* and *prove_model_ezkl.py* supports utils for calculating accuracy, performance degradation when converted to zk circuit, and checking time complexity for generating proof / proof verification. The results will come out after all proofs have been generated and verified. We observe performance degradation for EZKL from time-to-time, in which we conjecture this may be due to different weights in each train time. We plan to look deeper in the future.
