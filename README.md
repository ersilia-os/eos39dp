# Pharmacokinetics Profiler (PhaKinPro)

Pharmacokinetics Profiler (PhaKinPro) predicts the pharmacokinetic (PK) properties of drug candidates. It has been built using a manually curated database of 10.000 compounds with information for 12 PK endpoints. Each model provides a multi-classifier output for a single endpoint, along with a confidence estimate of the prediction and whether the query molecule is within the applicability domain of the model.

This model was incorporated on 2024-05-03.


## Information
### Identifiers
- **Ersilia Identifier:** `eos39dp`
- **Slug:** `phakinpro`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Property calculation or prediction`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `ADME`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `18`
- **Output Consistency:** `Fixed`
- **Interpretation:** A list of several ADME predictions

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| hs_15min | float | high | Probability of hepatic stability <=50% at 15 min |
| hs_30min | float | high | Probability of hepatic stability <=50% at 30 min |
| hs_60min | float | high | Probability of hepatic stability <=50% at 60 min |
| mhl_subcellular | float | high | Probability of sub-cellular hepatic Half-life <= 30 minutes |
| mhl_tissue | float | high | Probability of tissue hepatic Half-life <= 30 minutes |
| rc_01 | float | high | Probability of renal clearance below 0.10 ml/min/kg |
| rc_05 | float | high | Probability of renal clearance below 0.50 ml/min/kg |
| rc_1 | float | high | Probability of renal clearance below 1.00 ml/min/kg |
| bbb_permeability | float | high | Probability of permeating the blood-brain barrier |
| cns_activity | float | high | Probability of exhibiting central nervous system activity |

_10 of 18 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos39dp](https://hub.docker.com/r/ersiliaos/eos39dp)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos39dp.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos39dp.zip)

### Resource Consumption
- **Model Size (Mb):** `50`
- **Environment Size (Mb):** `2397`
- **Image Size (Mb):** `2332.36`

**Computational Performance (seconds):**
- 10 inputs: `28.5`
- 100 inputs: `18.39`
- 10000 inputs: `143.94`

### References
- **Source Code**: [https://github.com/molecularmodelinglab/PhaKinPro](https://github.com/molecularmodelinglab/PhaKinPro)
- **Publication**: [https://pubs.acs.org/doi/10.1021/acs.jmedchem.3c02446](https://pubs.acs.org/doi/10.1021/acs.jmedchem.3c02446)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2024`
- **Ersilia Contributor:** [sucksido](https://github.com/sucksido)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos39dp
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos39dp
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
