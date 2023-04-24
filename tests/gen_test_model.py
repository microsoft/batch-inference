# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch


class Mul(torch.nn.Module):
    def forward(self, x, y):
        res = torch.matmul(x, y)
        return res


x = torch.randn(2, 3, 3)
y = torch.randn(2, 3, 3)

model = Mul()

torch.onnx.export(
    model,  # model being run
    (x, y),  # model input (or a tuple for multiple inputs)
    "matmul.onnx",  # where to save the model (can be a file or file-like object)
    input_names=["x", "y"],  # the model's input names
    output_names=["res"],
    dynamic_axes={
        "x": {0: "batch_size"},  # variable length axes
        "y": {0: "batch_size"},
        "res": {0: "batch_size"},
    },
)  # the model's output names
