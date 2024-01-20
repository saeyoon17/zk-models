import json
import os

import ezkl
import ipdb
import torch
from torch.utils.data import DataLoader

from data import HeartFailureDataset
from models import LinearRegression

from train import collate_fn

"""Get checkpoints, test dataset"""

PATH = "./model_ckpt.pt"
ckpt = torch.load(PATH)
in_dim = 18
out_dim = 2
model = LinearRegression(in_dim=in_dim, out_dim=out_dim)
model.load_state_dict(ckpt["model_state_dict"])
test_data = HeartFailureDataset(split="test")
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, collate_fn=collate_fn)

"""EZKL configurations"""
model_path = os.path.join("network.onnx")
compiled_model_path = os.path.join("network.compiled")
pk_path = os.path.join("test.pk")
vk_path = os.path.join("test.vk")
settings_path = os.path.join("settings.json")

witness_path = os.path.join("witness.json")
data_path = os.path.join("input.json")

# TODO: Change this into test data
for feat, label in test_loader:
    # Flips the neural net into inference mode
    model.eval()
    # ipdb.set_trace()
    torch_out = model(feat)
    # Export the model
    torch.onnx.export(
        model,  # model being run
        feat,  # model input (or a tuple for multiple inputs)
        model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    data_array = ((feat).detach().numpy()).reshape([-1]).tolist()

    #    data = dict(input_data=[data_array])
    data = dict(
        input_shapes=[feat.shape],
        input_data=[data_array],
        output_data=[((torch_out).detach().numpy()).reshape([-1]).tolist()],
    )
    # ipdb.set_trace()
    # Serialize data into file:
    json.dump(data, open(data_path, "w"))

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "private"  # private by default
    py_run_args.variables = [("batch_size", 64)]

    res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
    assert res == True

    # calibration
    ezkl.calibrate_settings(data_path, model_path, settings_path, "resources")

    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True

    # srs path
    res = ezkl.get_srs(settings_path)

    # now generate the witness file
    # ipdb.set_trace()
    res = ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    # HERE WE SETUP THE CIRCUIT PARAMS
    # WE GOT KEYS
    # WE GOT CIRCUIT PARAMETERS
    # EVERYTHING ANYONE HAS EVER NEEDED FOR ZK

    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
    )

    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)
    # first of all, check this for the first batch only.

    # GENERATE A PROOF

    proof_path = os.path.join("test.pf")

    res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
    )

    print(res)
    assert os.path.isfile(proof_path)

    # VERIFY IT

    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
    )

    assert res == True
    print("verified")

    ipdb.set_trace()
