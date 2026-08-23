#!/usr/bin/env python3
from __future__ import annotations
import sys
import contract_hardening
import train
contract_hardening.install_finetuner(train,"tabular_classification")
if __name__=="__main__":sys.exit(train.main())
