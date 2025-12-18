# ODesign Inference Instructions

Follow the steps below to set up and run the inference pipeline.

## 1. Environment Setup
The environment configuration and dependencies are identical to **`odesign-unify`**. Please ensure you are using the same Conda environment or installation setup.

## 2. Install Checkpoints
Run the provided script to download the necessary model checkpoints:

```bash
bash ./ckpt/get_odesign_ckpt.sh
```

## 3. Prepare Input JSON
Create a JSON file to define your design task.

**Note:** Unlike the `odesign-unify` version, this workflow allows you to input an antibody framework sequence and mask the regions designated for design using hyphens (`-`).

### Example Configuration
```json
[
    {
        "name": "abtest",
        "antigen": "/data/qinghan/OFAntibody/rfantibody_targets/truncated/6oq5_tcdb_truncated.pdb",
        "hotspot": "A/538,A/151,A/152,A/148,A/539",
        "chains": [
            {
                "chain_type": "proteinChain",
                "im": "antibody",
                "sequence": "EVQLVESGGGLVQPGGSLRLSCAAS-YIHWVRQAPGKGLEWVARI-TRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR-WGQGTLVTVSS",
                "length": "6-7,6-7,9-15"
            },
            {
                "chain_type": "proteinChain",
                "im": "antibody",
                "sequence": "DIQMTQSPSSLSASVGDRVTITC-WYQQKPGKAPKLLIY-GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC-FGQGTKVEIK",
                "length": "6-7,6-7,9-15"
            },
            {
                "chain_type": "proteinChain",
                "im": "antigen",
                "sequence": "A/100-550"
            }
        ]
    }
]
```

### Parameter Description

| Parameter | Description |
| :--- | :--- |
| **name** | The name of the sample. |
| **antigen** | Path to the antigen structure file (`.cif` or `.pdb`). |
| **hotspot** | User-specified hotspot residues (e.g., `Chain/ResidueIndex`). |
| **chains** | Defines the components of the design (consistent with `odesign-unify`). |

**Detailed `chains` Configuration:**

* **`im` (Identity Mode):**
    * `antibody`: The model reads the sequence from the `sequence` field.
    * `antigen`: The model reads the structure from the file specified in the top-level `antigen` path.
* **`sequence`:**
    * **For Antigen:** Specify the residue range (e.g., `A/100-550`), same as `odesign-unify`.
    * **For Antibody:** Provide the framework sequence. Replace the CDRs (Complementarity-Determining Regions) to be designed with a hyphen (`-`).
* **`length`:**
    * *(Antibody only)* Specifies the target length range for each masked CDR.
    * The order corresponds to the hyphens in the `sequence` string (e.g., `6-7,6-7,9-15` corresponds to the first, second, and third `-`).

## 4. Download Data Dependencies
Before running the inference for the first time, you must download the required component files from Google Drive.

* **Files:** `components.v20240608.cif` and `components.v20240608.cif.rdkit_mol.pkl` from [Google Drive](https://drive.google.com/drive/folders/1wPmwIrC3G52q1JFY0RXY95tjKDl7YEln?usp=drive_link)
* **Destination:** Place these files into your specified `data_root_dir`.

## 5. Run Inference Script
Execute the inference script with the following parameters:

```bash
python ./scripts/inference.py \
    data_root_dir="$data_root_dir" \
    ckpt_root_dir="$ckpt_root_dir" \
    exp.input_json_path="$input_json_path" \
    exp.exp_name="$exp_name" \
    exp.seeds="$seeds" \
    exp.model.sample_diffusion.N_sample="$N_sample" \
    exp.num_workers="$num_workers"
```