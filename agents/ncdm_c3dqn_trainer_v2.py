"""Corrected unified runtime for Base and Set C3DQN-NCDM.

This module subclasses the legacy trainer to preserve its public training API while
replacing the model forwarding, replay batching, alpha update, AMP, checkpoint,
and candidate-pool logic with the Set-aware implementation.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
from contextlib import nullcontext
import random
import time

import torch
import torch.nn.functional as F
from torch import nn

from agents.ncdm_c3dqn_trainer import (
    BASE_ARCHITECTURE if False else C3DQNTransition,
)
