# Roadmap

The roadmap outlines major planned additions/changes to the code.

## Current Development

- **Masked Autoregressive Flows**: Currently being implemented in [maf](https://github.com/htjb/margarine/tree/maf)

## Planned Additions

- **Alternate Normalisation**: Implement standardization as an alternate to the existing gaussianization. Similar to [margarine_unbounded](https://github.com/mrosep/margarine_unbounded).
- **Evidence Calculations**: Integration of [floZ](https://arxiv.org/abs/2404.12294) style evidence calculations.
- **Neural Spline Flows**: Implementation of NSF for improved flexibility.
- **Conditional KDE and PNF**: Add conditional KDE and PNF alongside the existing conditional NICE and RealNVP.

## Future Exploration

- **Diffusion Models**: Explore addition of diffusion models as alternative density estimators.
