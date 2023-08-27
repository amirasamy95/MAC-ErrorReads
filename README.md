# MAC-ErrorReads
MAC-ErrorReads is a Machine Learning-Assisted Classifier for Filtering Erroneous NGS Reads. It converts the process of filtering erroneous NGS reads into a machine learning classification problem. MAC-ErrorReads learns a mapping function F that transforms the input features space X extracted from each sequencing read into a binary label classification Y of this read as (1) for erroneous read and (0) for correct one.

# Usage



# Tools
We are using different tools in our experment.

We used the wgsim (https://github.com/lh3/wgsim) simulator. Wgsim is a small tool for simulating sequence reads from a reference genome.

We also used the BWA aligner (https://github.com/lh3/bwa) to align the reads to the E. coli reference genome and extract their alignment scores from the resulting SAM files.

We also used Velvet assembler (https://github.com/dzerbino/velvet) to assemble the reads that are classified as error-free reads.

We also used QUAST (https://github.com/ablab/quast), the assembly evaluation tool, to evaluate the assembly results.

We are finally using Lighter (https://github.com/mourisl/Lighter), a KMER-based error correction method for genome sequencing data.We are using Lighter to correct the error reads.



# Data
We are using two different datasets. The first dataset used in our experiments is Escherichia coli str. K-12 substr. MG1655 (E.coli) with RefSeq accession entry NC_000913. The second dataset used in our experiments is the S. aureus genome from the GAUGE project.

We are using a simulated dataset to train our models. We used the wgsim simulator, which is included within the SAMtools for whole genome simulation.
To test our models, we are using simulated and real datasets.

The link to the simulated datasets used (https://drive.google.com/file/d/10nEAroKXB9uUEFjL8eZ6tle8wr5GryzX/view?usp=drive_link) to train and test our models

The first real dataset used to test our models is (E.coli) with accession number SRR625891, and the second real dataset used in our experiments is the S. aureus genome from the GAUGE project.

# License
This repository is under MIT license.

# Contact
Please do not hesitate to contact us (amira_samy@mans.edu.eg) if you have any comments, suggestions, or clarification requests regarding the study or if you would like to contribute to this resource.





