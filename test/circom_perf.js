const chai = require('chai');
const fs = require('fs');

const wasm_tester = require('circom_tester').wasm;

const F1Field = require('ffjavascript').F1Field;
const Scalar = require('ffjavascript').Scalar;
exports.p = Scalar.fromString('21888242871839275222246405745257275088548364400416034343698204186575808495617');
const Fr = new F1Field(exports.p);

const assert = chai.assert;

// const exec = require('await-exec');
describe('circom perf degradation', function () {
    this.timeout(100000000);
    describe('transformer', async () => {
        it('raw output', async () => {

            let INPUT = JSON.parse(fs.readFileSync('./circom_data/input_0.json'));
            let OUTPUT = JSON.parse(fs.readFileSync('./circom_data/output_0.json'));

            const circuit = await wasm_tester('./circom_circuits/linear_regression.circom');

            const witness = await circuit.calculateWitness(INPUT, true);

            assert(Fr.eq(Fr.e(witness[0]), Fr.e(1)));

            const scale = 1E-12;

            let predicted = [];
            for (var i = 0; i < OUTPUT['out'].length; i++) {
                predicted.push(parseFloat(Fr.toString(Fr.e(witness[i + 1]))) * scale);
            }

            let ape = 0;

            for (var i = 0; i < OUTPUT['out'].length; i++) {
                const actual = parseFloat(OUTPUT['out'][i]);
                console.log('actual', actual, 'predicted', predicted[i]);
                ape += Math.abs((predicted[i] - actual));
            }

            const mape = ape / OUTPUT['out'].length;

            console.log('mean absolute error ', mape);
        });
    });
});