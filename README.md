# Text Re-Identification (TRI)
This repository contains the code and data for the text re-identification attack presented in *B. Manzanares-Salor, D. Sánchez, P. Lison, Evaluating the disclosure risk of anonymized documents via a machine learning-based re-identification attack, Submitted, (2022)*. In addition, some results obtained during testing are provided. A first version of this project was presented in [*B. Manzanares-Salor, D. Sánchez, P. Lison, Automatic Evaluation of Disclosure Risks of Text Anonymization Methods, Privacy in Statistical Databases, (2022)*](https://link.springer.com/chapter/10.1007/978-3-031-13945-1_12).

The data were extracted from [this repository](https://github.com/fadiabdulf/automatic_text_anonymization), corresponding to the publication [*F. Hassan, D. Sanchez, J. Domingo-Ferrer, Utility-Preserving Privacy Protection of Textual Documents via Word Embeddings, IEEE Transactions on Knowledge and Data Engineering, (2021)*](https://ieeexplore.ieee.org/abstract/document/9419784). Some modifications have been performed, such as the addition of spaCy anonymization or a slight restructuring. The resulting data can be found in the [data.zip](data.zip) file.

The code is presented in the [TextRe-Identification.ipynb](TextRe-Identification.ipynb) notebook. The project can be run locally (using [Jupyter](https://jupyter.org/)) or in [Google Colab](https://colab.research.google.com/) (as it was done during the development and testing). If Colab is used, it is necessary to upload the contents of this repository to a [Google Drive](https://drive.google.com/) folder, so that Colab can access it. Specific instructions on how to run the code and installation of the required Python packages are included in the notebook.

# Project structure
```
Text Re-Identification
│   README.md                        # This README
│   results.csv                      # Some results obtained during testing
│   requirements.txt                 # Pip3 requirements file
│   TextRe-Identification.ipynb      # Code notebook
└───data.zip                         # Data folder compressed
    │   500_random_titles.txt        # Actors used in the paper as 500_random scenario
    │   500_filtered_titles.txt      # Actors used in the paper as 500_filtered scenario
    │   2000_filtered_titles.txt     # Actors used in the paper as 2000_filtered scenario
    └───train                        # Folder with train actors' files
    │   │   actorA.xml
    │   │   actorB.xml
    │   │   ...
    │
    └───eval                         # Folder with evaluation actors' files
        │   actorC.xml
        │   actorD.xml
        │   ...
```
