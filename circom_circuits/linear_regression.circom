pragma circom 2.0.0;

include "matElemMul.circom";
include "matElemSum.circom";

// matrix multiplication
template LinearRegression (m,n,p) {
    signal input a[m][n];
    signal input b[n][p];
    signal input bias[p];
    signal output out[m][p];

    component matElemMulComp[m][p];
    component matElemSumComp[m][p];

    for (var i=0; i < m; i++) {
        for (var j=0; j < p; j++) {
            matElemMulComp[i][j] = matElemMul(1,n);
            matElemSumComp[i][j] = matElemSum(1,n);
            for (var k=0; k < n; k++) {
                matElemMulComp[i][j].a[0][k] <== a[i][k];
                matElemMulComp[i][j].b[0][k] <== b[k][j];
            }
            for (var k=0; k < n; k++) {
                matElemSumComp[i][j].a[0][k] <== matElemMulComp[i][j].out[0][k];
            }
            out[i][j] <== matElemSumComp[i][j].out + bias[j];
        }
    }
}
//bs x input_dim x output_dim
//component main {public [a]} = LinearRegression(64,18,2);