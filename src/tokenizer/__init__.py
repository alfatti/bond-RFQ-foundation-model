# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .rfq_tokenizer import RFQTokenizer

# GPU-dependent tokenizers (cuDF/cuML) are imported lazily so the RFQ tokenizer
# and corpus tooling remain usable on a CPU-only dev box. On the GPU box these
# import normally; here they are skipped if RAPIDS is absent.
try:  # pragma: no cover - environment dependent
    from .financial_tokenizer import FinancialTabularTokenizer
    from .financial_pipeline import FinancialTokenizerPipeline
    from .pipeline import TokenizerPipeline
    from .base import BaseTokenizer
    from .fixed_vocab import FixedVocabTokenizer
    from .mapping import MappingTokenizer
    from .categorical_hash import CategoricalHashTokenizer
    from .numerical import NumericalTokenizerOptBin
    from .timedelta import TimeDeltaTokenizer

    _GPU_TOKENIZERS = [
        "FinancialTabularTokenizer",
        "FinancialTokenizerPipeline",
        "TokenizerPipeline",
        "BaseTokenizer",
        "FixedVocabTokenizer",
        "MappingTokenizer",
        "CategoricalHashTokenizer",
        "NumericalTokenizerOptBin",
        "TimeDeltaTokenizer",
    ]
except ImportError:  # RAPIDS not available (CPU dev box)
    _GPU_TOKENIZERS = []

__all__ = ["RFQTokenizer", *_GPU_TOKENIZERS]
