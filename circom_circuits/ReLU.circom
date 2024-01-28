pragma circom 2.0.0;

include "compconstant.circom";
include "comparators.circom";
include "aliascheck.circom";

// sign (source: https://github.com/socathie/circomlib-ml/blob/master/circuits/circomlib/sign.circom)
template Sign() {
    signal input in[254];
    signal output sign;

    component comp = CompConstant(10944121435919637611123202872628637544274182200208017171849102093287904247808);

    var i;

    for (i=0; i<254; i++) {
        comp.in[i] <== in[i];
    }

    sign <== comp.out;
}

//Positivity check (source: https://github.com/socathie/circomlib-ml/blob/master/circuits/util.circom)
template IsPositive() {
    signal input in;
    signal output out;

    component num2Bits = Num2Bits(254);
    num2Bits.in <== in;
    component sign = Sign();
    
    for (var i = 0; i < 254; i++) {
        sign.in[i] <== num2Bits.out[i];
    }

    out <== 1 - sign.sign;
}

//ReLU (source: https://github.com/socathie/circomlib-ml/blob/master/circuits/ReLU.circom)
template ReLU () {
    signal input in;
    signal output out;

    component isPositive = IsPositive();

    isPositive.in <== in;
    
    out <== in * isPositive.out;
}