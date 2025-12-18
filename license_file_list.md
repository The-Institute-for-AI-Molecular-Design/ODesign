# Apache License 2.0 File List

This document lists all Python files in the ODesign project that are licensed under Apache License 2.0.

**Copyright**: 2024 ByteDance and/or its affiliates  
**License**: Apache License 2.0  
**Total Files**: 70

---

## 📂 Core Model Files (10 files)

1. `src/model/odesign.py`
2. `src/model/modules/loss.py`
3. `src/model/modules/frames.py`
4. `src/model/modules/primitives.py`
5. `src/model/modules/pairformer.py`
6. `src/model/modules/generator.py`
7. `src/model/modules/head.py`
8. `src/model/modules/diffusion.py`
9. `src/model/modules/transformer.py`
10. `src/model/modules/embedders.py`

## 📊 Data Processing Files (17 files)

11. `src/data/dataset.py`
12. `src/data/dataloader.py`
13. `src/utils/data/msa_featurizer.py`
14. `src/utils/data/parser.py`
15. `src/utils/data/constraint_featurizer.py`
16. `src/utils/data/misc.py`
17. `src/utils/data/constants.py`
18. `src/utils/data/cropping.py`
19. `src/utils/data/data_pipeline.py`
20. `src/utils/data/featurizer.py`
21. `src/utils/data/msa_utils.py`
22. `src/utils/data/file_io.py`
23. `src/utils/data/ccd.py`
24. `src/utils/data/tokenizer.py`
25. `src/utils/data/substructure_perms.py`
26. `src/utils/data/geometry.py`
27. `src/utils/data/filter.py`

## 🔄 Permutation Files (7 files)

28. `src/utils/permutation/permutation.py`
29. `src/utils/permutation/utils.py`
30. `src/utils/permutation/atom_permutation.py`
31. `src/utils/permutation/chain_permutation/__init__.py`
32. `src/utils/permutation/chain_permutation/heuristic.py`
33. `src/utils/permutation/chain_permutation/utils.py`
34. `src/utils/permutation/chain_permutation/pocket_based_permutation.py`

## 🛠️ Model Utilities (3 files)

35. `src/utils/model/rmsd.py`
36. `src/utils/model/misc.py`
37. `src/utils/model/torch_utils.py`

## 🎓 Training & Inference Files (4 files)

38. `src/utils/train/lr_scheduler.py`
39. `src/utils/train/metrics.py`
40. `src/utils/train/distributed.py`
41. `src/utils/inference/dumper.py`

## 🔧 OpenFold Local Utilities (33 files)

### Core Utilities (7 files)

42. `src/utils/openfold_local/utils/tensor_utils.py`
43. `src/utils/openfold_local/utils/rigid_utils.py`
44. `src/utils/openfold_local/utils/precision_utils.py`
45. `src/utils/openfold_local/utils/all_atom_multimer.py`
46. `src/utils/openfold_local/utils/chunk_utils.py`
47. `src/utils/openfold_local/utils/checkpointing.py`
48. `src/utils/openfold_local/utils/feats.py`

### Geometry Utilities (6 files)

49. `src/utils/openfold_local/utils/geometry/__init__.py`
50. `src/utils/openfold_local/utils/geometry/rigid_matrix_vector.py`
51. `src/utils/openfold_local/utils/geometry/rotation_matrix.py`
52. `src/utils/openfold_local/utils/geometry/utils.py`
53. `src/utils/openfold_local/utils/geometry/vector.py`
54. `src/utils/openfold_local/utils/geometry/test_utils.py`

### Kernel (1 file)

55. `src/utils/openfold_local/utils/kernel/attention_core.py`

### Model Components (5 files)

56. `src/utils/openfold_local/model/triangular_multiplicative_update.py`
57. `src/utils/openfold_local/model/outer_product_mean.py`
58. `src/utils/openfold_local/model/dropout.py`
59. `src/utils/openfold_local/model/triangular_attention.py`
60. `src/utils/openfold_local/model/primitives.py`

### Data Processing (9 files)

61. `src/utils/openfold_local/data/msa_identifiers.py`
62. `src/utils/openfold_local/data/parsers.py`
63. `src/utils/openfold_local/data/msa_pairing.py`
64. `src/utils/openfold_local/data/errors.py`
65. `src/utils/openfold_local/data/templates.py`
66. `src/utils/openfold_local/data/data_transforms.py`
67. `src/utils/openfold_local/data/mmcif_parsing.py`
68. `src/utils/openfold_local/data/tools/utils.py`
69. `src/utils/openfold_local/data/tools/jackhmmer.py`

### Residue Constants (1 file)

70. `src/utils/openfold_local/np/residue_constants.py`

---

## 📋 Summary by Category

| Category | File Count |
|----------|------------|
| Core Model | 10 |
| Data Processing | 17 |
| Permutation | 7 |
| Model Utilities | 3 |
| Training & Inference | 4 |
| OpenFold Local | 33 |
| **Total** | **70** |

---

## 📝 License Information

All files listed above contain the following license header:

```python
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

---

## ✅ Commercial Use Compliance

To use these files in commercial products, you must:

1. ✅ **Include License**: Include a copy of the Apache License 2.0
2. ✅ **Copyright Notice**: Retain all copyright notices from the source files
3. ✅ **State Changes**: Document any modifications made to the files
4. ✅ **Include NOTICE**: If a NOTICE file exists, include it in distributions

---

**Last Updated**: 2024  
**Generated**: Automated scan of src/ directory  
**License**: Apache License 2.0

