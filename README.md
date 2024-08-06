# This is a forked version of the Olympus package
Please see the folder [`case_study_pwas`](https://github.com/mjzhu-p/olympus/tree/pwas_comp/case_studies/case_study_pwas/z_comparisonStudy) under ``case_studies`` for the detailed comparison studies reported in the paper "[_Discrete and mixed-variable experimental design with surrogate-based approach_](https://urldefense.com/v3/__https://chemrxiv.org/engage/chemrxiv/article-details/6626a713418a5379b0674df2__;!!PDiH4ENfjr2_Jw!C8gWZN1GmplOFNJc2FBxMi5gJxaT-VvJbeNeaq0kOX_WpczRONO5EQw0Qfseh-HHnwS48_McOz7EYB8TCjhzGguCluiS7A$)"

Please see the [main repository](https://github.com/MolChemML/ExpDesign) for the explanation of the case studies.
* See ["How to use this repository"](https://github.com/MolChemML/ExpDesign?tab=readme-ov-file#howto) for a step-by-step guide on relevant library installations.
* See ["Case studies"](https://github.com/MolChemML/ExpDesign?tab=readme-ov-file#casestudies) for details on which notebooks/scripts to run

**Note**:
* for **_EDBO_**, minor updates are required for the `bro.py` file to be able to specify the number of initial samples. See the specifics at https://github.com/mjzhu-p/edbo/tree/pwas_comp
* for **_Olympus_** package modified here, some updates are only needed for Windows systems (see the commit `2615b001fcfd9754a03f5a4b43b69c30afd4d993` adopted from [@felix-s-k](https://github.com/aspuru-guzik-group/olympus/pull/34) , you can revert changes in this commit if you are using a Linux system).

&nbsp;

```
@article{ExpDesign2024,
  title={Discrete and mixed-variable experimental design with surrogate-based approach},
  author={Zhu, Mengjia and Mroz, Austin and Gui, Lingfeng and Jelfs, Kim and Bemporad, Alberto and del Río Chanona, Ehecatl Antonio and Lee, Ye Seol},
  journal={ChemRxiv preprint  doi:10.26434/chemrxiv-2024-h37x4},
  year={2024}
}
```
