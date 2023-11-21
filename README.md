# MAC-ErrorReads
MAC-ErrorReads is a machine learning-assisted classifier for filtering erroneous NGS reads. It converts the process of filtering erroneous NGS reads into a machine learning classification problem. MAC-ErrorReads learns a mapping function F that transforms the input feature space X extracted from each sequencing read into a binary label classification Y of this read as (1) for an erroneous read and (0) for a correct one.

# Usage
We are using MAC-ErrorReads to classify the sequencing reads as correct or erroneous reads. Before the training step, we extract all k-mers from all reads and compute the TF-IDF for all k-mers as a feature extraction to represent a set of training features that feed into different machine algorithms for classifying each read as erroneous or error-free.

trainning step:
We train our models using simulated datasets.The first genome used in the training process of the MAC-ErrorReads system is Escherichia coli str. K-12 substr. MG1655 (E. coli) with RefSeq accession entry NC_000913. Machine learning models were trained using various k-mer sizes (7, 9, 11, 13, and 15) on a dataset comprising 400,000 correctly reads labeled with 0 and 400,000 erroneous reads labeled with 1. We used a wgsim simulator, which is included within the SAMtools for whole genome simulation, to generate the required number of N=800000 paired-end reads with L=300, and the error rate e=0 for correct reads, and e=1 for erroneous reads. We split the data into training (600000 reads) and testing (200000 reads).

The second genome used in the training process is Staphylococcus aureus (S. aureus) from GAGE, the genome size is 2903081bp and using C=30X, and L=101bp, the total number of paired-end reads used to cover the genome is N≤862301. We trained machine learning models using 400000 correct reads labeled with 1 and 400000 erroneous reads labeled with 0. We split the data into training (600000 reads) and testing (200000 reads). We used k-mer size equals 15.

The third genome used in the training process is Human Chromosome 14 (H. Chr14) from GAGE, the genome size is 88289540 bp , and using C=30X, and L=101bp, the total number of paired-end reads used to cover the genome is N≤8828954. We trained the NB machine learning model with k=11 using 500000 correct reads labeled with 0 and 500000 erroneous reads labeled with 1. We split the data into training (700000 reads) and testing (300000 reads). The NB model is chosen since it demonstrates consistent and strong performance in the classification and assembly results of the previous experiments.

Evaluate step:

To evaluate the efficacy of our trained models, we employed reads obtained from simulated data and real sequencing experiments that corresponded to the previously trained reference genomes.
The first E. coli dataset used to test the performance of SVM, RF, LR, NB, and XGBoost is the simulated data set with the total number of paired-end reads is 200000.To evaluate the effectiveness of the E. coli trained model on reads from real sequencing experiments, we utilized a real dataset with accession number SRR625891. 

The second  S. aureus dataset used to test the performance of SVM, RF, LR, NB, and XGBoost is the simulated data set with the total number of paired-end reads is 200000.To evaluate the effectiveness of the S. aureus trained model on reads from real sequencing experiments, we utilized a real dataset from GAGE for the same genome. 

The third  Human Chromosome 14 (H. Chr14) dataset used to test the performance of  NB  is the simulated data set with the total number of paired-end reads is 300000.To evaluate the effectiveness of the (H. Chr14) trained model on reads from real sequencing experiments, we utilized a real dataset from GAGE for the same genome. 

The real reads in the sequencing experiments lacked explicit labels, and their accuracy levels were unknown beforehand.

To address this issue in the first dataset (E. coli), we proceeded with aligning the reads to their reference genome through the use of one of the NGS aligners (i.e., BWA). Subsequently, we extracted the computed alignment score for each read. By comparing the predicted labels generated by our trained machine learning models with the labels inferred from the alignment scores, we could effectively evaluate the performance of these models.

To address this issue in the second dataset (S. aureus), we assembled the reads generated after testing the data using Velvet assembler. After the assembly results are evaluated by one of the assembly evaluation tools called QUAST.We then used Lighter to correct the reads that were classified as errors and assembled the files that contained correct reads and errors that were corrected by Lighter.
We also computing the alignment rates. the alignment statistics computed using Bowtie2.

To address this issue in the third dataset (H. Chr14), H. Chr14 reference genome using Bowtie 2, and alignment statistics were computed. 
# Tools
We are using different tools in our experment.

We used the wgsim (https://github.com/lh3/wgsim) simulator. Wgsim is a small tool for simulating sequence reads from a reference genome.

We also used the BWA aligner (https://github.com/lh3/bwa) to align the reads to the E. coli reference genome and extract their alignment scores from the resulting SAM files.

We also used Velvet assembler (https://github.com/dzerbino/velvet) to assemble the reads that are classified as error-free reads.

We also used QUAST (https://github.com/ablab/quast), the assembly evaluation tool, to evaluate the assembly results.

We are finally using Lighter (https://github.com/mourisl/Lighter), a KMER-based error correction method for genome sequencing data.We are using Lighter to correct the error reads.



# Data
We are using three different datasets. The first dataset used in our experiments is Escherichia coli str. K-12 substr. MG1655 (E.coli) with RefSeq accession entry NC_000913. The second dataset used in our experiments is the S. aureus genome from the GAUGE project.

We are using a simulated dataset to train our models. We used the wgsim simulator, which is included within the SAMtools for whole genome simulation.
To test our models, we are using simulated and real datasets.

The link to the simulated datasets used (https://drive.google.com/file/d/10nEAroKXB9uUEFjL8eZ6tle8wr5GryzX/view?usp=drive_link) to train and test our models

The first real dataset used to test our models is (E.coli) with accession number SRR625891, and the second real dataset used in our experiments is the S. aureus genome from the GAUGE project.

# License
This repository is under MIT license.

# Contact
Please do not hesitate to contact us (amira_samy@mans.edu.eg) if you have any comments, suggestions, or clarification requests regarding the study or if you would like to contribute to this resource.





