pragma circom 2.0.0;

include "matElemMul.circom";
include "matElemSum.circom";
include "ReLU.circom";
include "linear_regression.circom";


// MLP
template MLP (batch_size,in,hidden,out) {
    signal input batch_in[batch_size][in];
    signal input weight1[in][hidden];
    signal input weight2[hidden][hidden];
    signal input weight3[hidden][out];
    signal input bias1[hidden];
    signal input bias2[hidden];
    signal input bias3[out];
    signal output batch_out[batch_size][out];

    component linear1 = LinearRegression(batch_size, in, hidden);
    component linear2 = LinearRegression(batch_size, hidden, hidden);
    component linear3 = LinearRegression(batch_size, hidden, out);
    component relu1[batch_size][hidden];
    for (var i=0; i < batch_size; i++) {
        for (var j=0; j < hidden; j++){
            relu1[i][j] = ReLU();
        }
    }
    component relu2[batch_size][hidden];
    for (var i=0; i < batch_size; i++) {
        for (var j=0; j < hidden; j++){
            relu2[i][j] = ReLU();
        }
    }
    for (var i = 0; i < batch_size; i++) {
        for (var j = 0; j < in; j++) {
            linear1.a[i][j] <== batch_in[i][j];
        }
    }
    for (var i = 0; i < in; i++) {
        for (var j=0; j < hidden; j++) {
            linear1.b[i][j] <== weight1[i][j];
        }
    }
    for (var i=0; i < hidden; i++) {
        linear1.bias[i] <== bias1[i];
    }

    for (var i=0; i < batch_size; i++) {
        for (var j=0; j < hidden; j++){
            relu1[i][j].in <== linear1.out[i][j];
        }
    }

    for (var i=0; i < batch_size; i++) {
        for (var j=0; j < hidden; j++) {
            linear2.a[i][j] <== relu1[i][j].out;
        }
    }
    for (var i=0; i < hidden; i++) {
        for (var j=0; j < hidden; j++) {
            linear2.b[i][j] <== weight2[i][j];
        }
    }
    for (var i=0; i < hidden; i++) {
        linear2.bias[i] <== bias2[i];
    }

    for (var i=0; i < batch_size; i++) {
        for (var j=0; j < hidden; j++){
            relu2[i][j].in <== linear2.out[i][j];
        }
    }

    for (var i=0; i < batch_size; i++) {
        for (var j=0; j < hidden; j++) {
            linear3.a[i][j] <== relu2[i][j].out;
        }
    }
    for (var i=0; i < hidden; i++) {
        for (var j=0; j < out; j++) {
            linear3.b[i][j] <== weight3[i][j];
        }
    }
    for (var i=0; i < out; i++) {
        linear3.bias[i] <== bias3[i];
    }
    for (var i=0; i < batch_size; i++) {
        for (var j=0; j < out; j++){
            batch_out[i][j] <== linear3.out[i][j];
        }
    }
}
//bs x input_dim x output_dim
component main {public [batch_in]} = MLP(16,18,16, 2);