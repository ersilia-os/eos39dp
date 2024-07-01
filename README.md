# Pharmacokinetics Profiler (PhaKinPro)

Pharmacokinetics Profiler (PhaKinPro) is a recently developed web-based tool that helps predict the pharmacokinetic (PK) properties of drug candidates. In essence, it assists scientists in determining how a drug will behave within the body.
Pharmacokinetics refers to the processes by which a drug is absorbed, distributed, metabolized, and excreted (ADME). Understanding these processes is critical in drug development, as they can affect a drug's efficacy and safety. For example, a drug that is rapidly metabolized may not be effective in the body, while a drug that is slowly excreted may accumulate to toxic levels.

## Identifiers

* EOS model ID: `eos39dp`
* Slug: `phakinpro`

## Characteristics

* Input: `Compound`
* Input Shape: `Single`
* Task: `Regression`
* Output: `Compound`
* Output Type: `Float`
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