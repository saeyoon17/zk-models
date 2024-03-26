pragma circom 2.0.0;

include "comparators.circom";

template Sum(n) {
  signal input in[n];
  signal output out;

  var lc = 0;
  for (var i = 0; i < n; i++) {
    lc += in[i];
  }
  out <== lc;
}

template kMeans (batch_size){

    // Compute e0
    signal input e0[18];
    signal input e1[18];
    signal input data[batch_size][18];
    signal output out[batch_size];

    signal e0_diff[batch_size][18];
    signal e1_diff[batch_size][18];
    for (var i0 = 0; i0 < batch_size; i0 ++){
        for (var i1 = 0; i1 < 18; i1++){
            e0_diff[i0][i1] <== (data[i0][i1] - e0[i1])**2;
            e1_diff[i0][i1] <== (data[i0][i1] - e1[i1])**2;
        }
    }

    component e0_score[batch_size];
    component e1_score[batch_size];

    for (var i0 = 0; i0 < batch_size; i0 ++){
        e0_score[i0] = Sum(18);
        e1_score[i0] = Sum(18);
    }

    for (var i0 = 0; i0 < batch_size; i0 ++){
        e0_score[i0].in <== e0_diff[i0];
        e1_score[i0].in <== e1_diff[i0];
    }

    component lt[batch_size];
    for (var i0 = 0; i0 < batch_size; i0 ++){
        lt[i0] = LessThan(64);
    }

    for (var i0 = 0; i0 < batch_size; i0 ++){
        lt[i0].in[0] <== e0_score[i0].out;
        lt[i0].in[1] <== e1_score[i0].out;
    }

    for (var i0 = 0; i0 < batch_size; i0 ++){
        out[i0] <== lt[i0].out;
    }


}

// Define component inputs and outputs
component main {public [data]}= kMeans(16);
