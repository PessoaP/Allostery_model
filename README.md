# Code for ''Allostery Beyond Amplification: Temporal Regulation of Signaling Information''

This repository contains the Python code used to generate the figures in the manuscript  
**_Allostery Beyond Amplification: Temporal Regulation of Signaling Information_**.
Available as a preprint on [arXiv](https://arxiv.org/abs/2601.01850)

The code implements a chemical master equation (CME) formulation of the allosteric signaling model in the figure below and solves it numerically using sparse transition-rate matrices

<div style="background-color:white; display:inline-block; padding:10px;">
  <img src="http://labpresse.com/wp-content/uploads/2026/01/Pessoa_Allo-1.png" width="750"/>
</div>

---

## Repository Structure

**`allostery_fixed.py`**  
Computes steady-state solutions of the CME for fixed kinetic parameters while sweeping the substrate production rate. This script produces the results shown in **Fig. 2** of the manuscript, including steady-state mutual information and mean molecular copy numbers.

**`allostery_sweep.py`**  
Performs parameter sweeps over K-type and V-type allosteric ratios at fixed substrate input. This script is used to generate **Fig. 3**, illustrating how information transmission varies independently of product abundance.

**`allostery_variable.py`**  
Solves the CME with a time-dependent substrate production rate, implementing pulsed (on/off) input protocols. This script generates the time-resolved results shown in **Fig. 4**, including transient information bursts.

---

## Usage

Each script can be run independently:

```bash
python allostery_fixed.py
python allostery_sweep.py
python allostery_variable.py

```

All model parameters are defined within the scripts and correspond directly to those reported in the manuscript.


---

## Citation

If you find this work useful, we appreciate the citation. Here's the BibTeX:
```
@misc{pessoa2025allostery,
      title={Allostery Beyond Amplification: Temporal Regulation of Signaling Information }, 
      author={Pedro Pessoa and Steve Pressé and S Banu Ozkan},  
      year={2026},
      eprint={2601.01850},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2601.01850}, 
}
```
