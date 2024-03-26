# zk-models
This repository aims to provide pipelines for trainig neural networks and proving them by transforming networks to zero-knowledge circuits.
Current implementation supports transforming multi-layer perceptron (MLP) with flexible number of layers to zk circuit.
You can choose to convert neural network to zk circuit by using Circom or EZKL.

## Installing packages
You can install necessary packages by running:
```bash
pip install -r requirements.txt
```
# Training neural network.
*train_mlp.py* provides pipeline for training heart failure prediction model. You can flexibly select the hidden vector dimension and number of hidden layers inside the file.
You can train your model by running:
```bash
python train_mlp.py
```
The dataset is contained in the repository under name *heart.csv*. 
Specific details about the dataset can be found [here](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

After the training has finished, you can find your model checkpoint under *data* directory.

## Converting neural network using Circom
For circom proof generation, we use Groth16 backend. Therefore, you first need to go through Groth16 specific setups.
They only need to be run once. After generating the keys, they can be reused in later proving protocol.
They are: 

```bash
cd circom_data
circom ../circom_circuits/mlp.circom --r1cs --wasm --sym
snarkjs powersoftau new bn128 19 pot19_0000.ptau -v
snarkjs powersoftau contribute pot19_0000.ptau pot19_0001.ptau --name="First contribution" -v
snarkjs powersoftau prepare phase2 pot19_0001.ptau pot19_final.ptau -v
snarkjs groth16 setup mlp.r1cs pot19_final.ptau proof0.key
snarkjs zkey contribute proof0.key proof01.key --name="your name" -v
snarkjs zkey export verificationkey proof01.key verification_key.json
```
*or*, you can run *key-gen.sh* inside circom_data/ directory. Make sure to change proving constant depending on the circuit size you are trying to prove.  After thant, you can run 
```bash
cd ..
python prove_model_circom.py
```
for proving your model. We note that current code supports proof generation for MLP. For linear regression, you may want to change file names and proxy calculation for checking performance degradation when neural network gets transformed into zk circuit. We are planning to modify the code so that user can switch between models with more flexibility. 

## Converting neural network using EZKL
After training your model, you can generate proof by running:
```bash
python prove_model_ezkl.py
```

# Training decision tree.
*train_decision_tree.py* trains scikit-learn decision tree for heart failure prediction. You can train decision tree by running:
```bash
python train_decision_tree.py
```
And then, you can prove decision tree with EZKL output by running: 
```bash
python prove_dt_ezkl.py
```

*zk-models* repository currently only supports proving decision tree with EZKL. Yet, we do provide utils for obtaining time complexity for proving decision tree [here](https://github.com/saeyoon17/ZKDT). Details can be found in the ZKDT forked repository.

# Training K-Means Clustering
*train_kmeans.py* performs scikit-learn K-Means Clustering for heart failure prediction. You can run clustering by:
```basg
python train_kmeans.py
```
And then, you can prove k-means clustering based classification output with EZKL by running: 
```bash
python prove_kmeans_ezkl.py
```
Also, you can prove classification with circom by running:
```bash
cd circom_circuits
bash key-gen.sh
cd ..
python prove_kmeans_circom.py
```
Please make sure to change circom circuit inside key-gen.sh to kmeans.circom since this is also used for proving MLP.
