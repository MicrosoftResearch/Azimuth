# Azimuth
##### Machine Learning-Based Predictive Modelling of CRISPR/Cas9 guide efficiency.
[![Travis](https://img.shields.io/travis/MicrosoftResearch/Azimuth.svg)](https://travis-ci.org/MicrosoftResearch/Azimuth) [![PyPI](https://img.shields.io/pypi/v/azimuth.svg)](https://pypi.python.org/pypi/azimuth) [![PyPI](https://img.shields.io/pypi/l/azimuth.svg)]()

The CRISPR/Cas9 system provides state-of-the art genome editing capabilities. However, several facets of this system are under investigation for further characterization and optimization. One in particular is the choice of guide RNA that directs Cas9 to target DNA: given that one would like to target the protein-coding region of a gene, hundreds of guides satisfy the constraints of the CRISPR/Cas9 Protospacer Adjacent Motif sequence. However, only some of these guides efficiently target DNA to generate gene knockouts. One could laboriously and systematically enumerate all possible guides for all possible genes and thereby derive a dictionary of efficient guides, however, such a process would be costly, time-consuming, and ultimately not practically feasible. Instead, one can (1) enumerate all possible guides over each of some smaller set of genes, and then test these experimentally by measuring the knockout capabilities of each guide, (2) thereby assemble a training data set with which one can "learn", by way of predictive machine learning models, which guides tend to perform well and which do not, (3) use this learned model to generalize the guide efficiency for genes not in the training data set. In particular, by deriving a large set of possible predictive features consisting of both guide and gene characteristics, one can elicit those characteristics that define guide-gene pairs in an abstract manner, enabling generalizing beyond those specific guides and genes, and in particular, for genes which we have never attempted to knock out and therefore have no experimental evidence. Based on such a set of experiments, we present a state-of-the art predictive approach to modeling which RNA guides will effectively perform a gene knockout by way of the CRISPR/Cas9 system. We demonstrate which features are critical for prediction (e.g., nucleotide identity), which are helpful (e.g., thermodynamics), and which are redundant (e.g., microhomology); then we combine our insights of useful features with exploration of different model classes, settling on one model which performs best (gradient-boosted regression trees). Finally, we elucidate which measures should be used for evaluating these models in such a context.

See our [**official project page**](https://www.microsoft.com/en-us/research/project/crispr/) for more detail.

#### Publications

Please cite this paper if using our predictive model:

**John G. Doench**\*, **Nicolo Fusi**\*, Meagan Sullender\*, Mudra Hegde\*, Emma W. Vaimberg\*, Katherine F. Donovan, Ian Smith, Zuzana Tothova, Craig Wilen , Robert Orchard, Herbert W. Virgin, **Jennifer Listgarten**\*, **David E. Root**.
[**Optimized sgRNA design to maximize activity and minimize off-target effects for genetic screens with CRISPR-Cas9**](https://doi.org/10.1038/nbt.3437). *Nature Biotechnology*, 2016.
(\* = equal contributions, **corresponding author**)

#### Official Releases

To view all the official releases of the Azimuth package, click on the "releases" tab above or follow [this link](https://github.com/MicrosoftResearch/Azimuth/releases).


#### Installation (python package)

Before installing Azimuth, we recommend downloading and installing [Anaconda](https://www.continuum.io/downloads).

Azimuth is available from the python package index. From the command prompt, type:

```shell
pip install azimuth
```

Alternatively, if you want access to the code, you can clone this repository.

To run our unit tests, navigate to the main Azimuth directory, and then from the command prompt, type
```shell
nosetests
```
If these pass, you will see "OK" as the last printout.

**If you prefer not to install python packages or download any code, you can use our model as a web service.** Instructions on how to do so are [HERE](https://www.microsoft.com/en-us/research/project/crispr/)

#### Getting started

From python, you can get predictions from our model by running:

```python
import azimuth.model_comparison

azimuth.model_comparison.predict(GUIDE, CUT_POSITION, PERCENT_PEPTIDE)[0]
```
where GUIDE, PERCENT_PEPTIDE and CUT_POSITION are numpy arrays.

**Note:** if CUT_POSITION and PERCENT_PEPTIDE are not provided or provided as `None`, a
separate model will be used that does not consider protein target site information.

#### Usage Example

```python
import azimuth.model_comparison
import numpy as np

sequences = np.array(['ACAGCTGATCTCCAGATATGACCATGGGTT', 'CAGCTGATCTCCAGATATGACCATGGGTTT', 'CCAGAAGTTTGAGCCACAAACCCATGGTCA'])
amino_acid_cut_positions = np.array([2, 2, 4])
percent_peptides = np.array([0.18, 0.18, 0.35])
predictions = azimuth.model_comparison.predict(sequences, amino_acid_cut_positions, percent_peptides)

for i, prediction in enumerate(predictions):
    print sequences[i], prediction
```

Output:
```
No model file specified, using V3_model_full
ACAGCTGATCTCCAGATATGACCATGGGTT 0.672298196907
CAGCTGATCTCCAGATATGACCATGGGTTT 0.687944237021
CCAGAAGTTTGAGCCACAAACCCATGGTCA 0.659245390401
```

#### Note about Azimuth scores

Although the data used for training were in the range 0.0 to 1.0, the predictions made by the final model are not explicitly normalized, so it is possible for Azimuth to make predictions outside of this range. We expect this to be somewhat rare, and it is reasonable to set these values to the closest part of the range \[0.0, 1.0\] (i.e. negative values to 0 and values greater than 1 to 1.0) if it is easier for your purposes.

#### Generating new model .pickle files

Sometimes the pre-computed .pickle files in the saved_models directory are incompatible with different versions of scikitlearn. You can re-train the files saved_models/V3_model_full.pickle and saved_models/V3_model_nopos.pickle by running the command python model_comparison.py (which will overwrite the saved models). You can check that the resulting models match the models we precomputed by running python test_saved_models.py within the directory tests.

#### Contacting us

You can submit bug reports using the GitHub issue tracker. If you have any other questions, please contact us at crispr@lists.research.microsoft.com.


