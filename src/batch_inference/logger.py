# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import sys

logger = logging.getLogger("batch-inference")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d:%(thread)d - %(levelname)s - %(message)s",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
