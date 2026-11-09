# APPLE-GO

This repository contains the core PyTorch implementation of the APPLE-GO forest canopy reflectance model.

## Paper

**APPLE-GO: Modeling high-spatial resolution forest canopy reflectance with effect of Adjacent Pixels using Path Length Extended Geometric Optical theory**, *Remote Sensing of Environment* **331** (2025), 115043. [Paper](https://www.sciencedirect.com/science/article/pii/S003442572500447X) · [DOI: 10.1016/j.rse.2025.115043](https://doi.org/10.1016/j.rse.2025.115043)

APPLE-GO models the bidirectional reflectance factor (BRF) of forest canopies at high spatial resolution while accounting for shading and cross-radiation from adjacent pixels. The paper uses a two-dimensional path length distribution (2-PLD) to calculate component area fractions, shading factors to describe the effect of neighboring trees on single scattering, and spectral invariant theory with an eight-neighbor convolution to estimate multiple scattering. The model was evaluated against LESS simulations and satellite observations.

## Repository structure

```text
APPLE-GO/
├── README.md
└── code/
    ├── model_Class_BRFmodel.py   # BRFModel entry point and input assembly
    ├── Class_Single_k.py         # Geometric quantities for single scattering
    ├── Class_Ac_Ag_big.py        # Geometry over the neighboring area
    ├── Class_Ac_nn.py            # Canopy geometry
    ├── Class_Ag_nn.py            # Ground geometry
    ├── Class_At_nn.py            # Crown-top geometry
    ├── Class_Az_nn.py            # Vertical geometry
    ├── Class_Edge_nn.py          # Canopy edge detection
    ├── Class_cal_k.py            # Geometric coefficient assembly
    ├── Class_cal_LAI.py          # LAI calculation
    ├── Class_cal_SF.py           # Leaf scattering factor
    ├── Class_p_lc.py            # Canopy-leaf probability and low-SZA correction
    ├── Class_p_lt.py            # Crown-top probability
    ├── Class_SingleBRF.py       # Single-scattering BRF
    ├── Class_MultiBRF.py        # Multiple-scattering BRF
    ├── Class_BRF_nn.py          # Single- and multiple-scattering combination
    ├── Class_Upsample.py        # Parameter upsampling
    └── read_and_write_tiff.py   # GeoTIFF I/O
```

## Inputs and output

By default, `BRFModel` reads `input/CHM.tif`, `input/PATH1.tif`, and `input/PATH2.tif`. Input rasters and a complete batch-running script are not included. The code requires PyTorch, NumPy, and the GDAL Python bindings (`osgeo`).

`BRFModel.forward()` returns a tensor shaped `[bands, rows, columns]` containing the total BRF with adjacent-pixel effects. The current implementation also includes a configurable low solar zenith angle (SZA) hotspot correction in `Class_p_lc.py`: it is fully applied at SZA ≤ 10°, transitions smoothly between 10° and 20°, and uses the original expression at SZA ≥ 20°. Set `hotspot_full_sza` and `hotspot_end_sza` to change these thresholds.
