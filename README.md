# MAC-ErrorReads
MAC-ErrorReads is a Machine Learning-Assisted Classifier for Filtering Erroneous NGS Reads. It converts the process of filtering erroneous NGS reads into a machine learning classification problem. MAC-ErrorReads learns a mapping function F that transforms the input features space X extracted from each sequencing read into a binary label classification Y of this read as (1) for erroneous read and (0) for correct one.
# Usage


# Tools

# Data
we are using two different datasets.The first dataset used in our experiments is Escherichia coli str. K-12 substr. MG1655 (E.coli) with RefSeq accession entry NC_000913. The second  dataset used in our experiments is S.aureus genome from the GAGE project. 
we are using simulated dataset to train our models dataset comprising 400,000 correctly reads labeled with 0 and 400,000 erroneous reads labeled with 1. we used a wgsim simulator, which is included within the SAMtools for whole genome simulation, to generate the required number of N=800000 paired-end reads in both datasets.
the link of the simulated datasets (https://drive.google.com/file/d/10nEAroKXB9uUEFjL8eZ6tle8wr5GryzX/view?usp=drive_link).


# License
This repository is under MIT license.

# Contact
Please do not hesitate to contact us (amira_samy@mans.edu.eg) if you have any comments, suggestions, or clarification requests regarding the study or if you would like to contribute to this resource.





