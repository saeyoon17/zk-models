circom ../circom_circuits/mlp_l3_d4.circom --r1cs --wasm --sym
snarkjs powersoftau new bn128 12 pot12_0000.ptau -v
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First contribution" -v <<EOF
asdf
EOF

snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v
snarkjs groth16 setup mlp.r1cs pot12_final.ptau proof0.key
snarkjs zkey contribute proof0.key proof01.key --name="your name" -v <<EOF
asdf
EOF
snarkjs zkey export verificationkey proof01.key verification_key.json