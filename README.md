# Pharmacokinetics Profiler (PhaKinPro)

Pharmacokinetics Profiler (PhaKinPro) predicts the pharmacokinetic (PK) properties of drug candidates. It has been built using a manually curated database of 10.000 compounds with information for 12 PK endpoints. Each model provides a multi-classifier output for a single endpoint, along with a confidence estimate of the prediction and whether the query molecule is within the applicability domain of the model.

## Identifiers

* EOS model ID: `eos39dp`
* Slug: `phakinpro`

## Characteristics

* Input: `Compound`
* Input Shape: `Single`
* Task: `Classification`
* Output: `Probability`
* Output Type: `String`
* Output Shape: `List`
* Interpretation: A list of several ADME predictions

## References

* [Publication](https://pubs.acs.org/doi/10.1021/acs.jmedchem.3c02446)
* [Source Code](https://github.com/molecularmodelinglab/PhaKinPro)
* Ersilia contributor: [sucksido](https://github.com/sucksido)

## Ersilia model URLs
* [GitHub](https://github.com/ersilia-os/eos39dp)
* [AWS S3](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos39dp.zip)
* [DockerHub](https://hub.docker.com/r/ersiliaos/eos39dp) (AMD64, ARM64)

## Citation

If you use this model, please cite the [original authors](https://pubs.acs.org/doi/10.1021/acs.jmedchem.3c02446) of the model and the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license. The model contained within this package is licensed under a MIT license.

Notice: Ersilia grants access to these models 'as is' provided by the original authors, please refer to the original code repository and/or publication if you use the model in your research.

## About Us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission!