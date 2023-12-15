# MAC-ErrorReads
<p align="justify"> MAC-ErrorReads is a machine-learning-assisted classifier for filtering erroneous NGS reads. It converts the process of filtering erroneous NGS reads into a machine learning classification problem. MAC-ErrorReads learns a mapping function F that transforms the input feature space X extracted from each sequencing read into a binary label classification Y of this read, as (1) for an erroneous read and (0) for a correct one. </p>

# Usage
<p align="justify"> We are using MAC-ErrorReads to classify the sequencing reads as correct or erroneous reads. Before the training step, we extract all k-mers from all reads and compute the TF-IDF for all k-mers as a feature extraction to represent a set of training features that feed into different machine algorithms for classifying each read as erroneous or error-free.</p>

**trainning step:**

<p align="justify"> We train our models using simulated datasets. The **first** genome used in the training process of the MAC-ErrorReads system is **Escherichia coli str**. K-12 substr. MG1655 (E. coli) with RefSeq accession entry NC_000913. Machine learning models were trained using various k-mer sizes (7, 9, 11, 13, and 15) on a dataset comprising 400,000 correctly read reads labelled with 0 and 400,000 erroneous reads labelled with 1. We used a wgsim simulator, which is included within the SAMtools for whole genome simulation, to generate the required number of N=800000 paired-end reads with L=300 and the error rate e=0 for correct reads and e=1 for erroneous reads. We split the data into training (600,000 reads) and testing (20,0000 reads). </p>

<p align="justify"> The **second** genome used in the training process is **Staphylococcus aureus** (S. aureus) from GAGE; the genome size is 2903081 bp, and using C = 30X and L = 101 bp, the total number of paired-end reads used to cover the genome is N≤862301. We trained machine learning models using 400000 correct reads labelled with 1 and 400000 erroneous reads labelled with 0. We split the data into training (600,000 reads) and testing (20,0000 reads). We used a k-mer size equal to 15. </p>

<p align="justify"> The **third** genome used in the training process is **Human Chromosome 14** (H. Chr14) from GAGE. The genome size is 88289540 bp, and using C = 30X and L = 101 bp, the total number of paired-end reads used to cover the genome is N≤8828954. We trained the NB machine learning model with k = 11 using 500000 correct reads labelled with 0 and 500000 erroneous reads labelled with 1. We split the data into training (700,000 reads) and testing (30,0000 reads). The NB model is chosen since it demonstrates consistent and strong performance in the classification and assembly results of the previous experiments. </p>

<p align="justify"> The **fourth** genome used in the training process is **Arabidopsis thaliana chromosome 1** with RefSeq accession entry NC_003070.9. The genome size is 119146348bp, and using C=30X, and L=250bp, the total number of paired-end reads used to cover the genome is N≤14297562. We trained the NB machine learning model with =11 using 200000 correct reads labeled with 0 and 200000 erroneous reads labeled with 1. We split the data into training (300000reads) and testing (100000reads).The NB model is chosen since it demonstrates consistent and strong performance in the classification and assembly results of the previous experiments. </p>

<p align="justify"> The **Fifth** genome used in the training process is **Metriaclima zebra** with RefSeq accession entry GCF_000238955.4, the genome size is 957468680bp, and using C=30X, and L=101bp, the total number of paired-end reads used to cover the genome is N≤284396638. We trained the NB machine learning model with =11 using 500000 correct reads labeled with 0 and 500000 erroneous reads labeled with 1. We split the data into training (700000reads) and testing (300000reads). The NB model is chosen since it demonstrates consistent and strong performance in the classification and assembly results of the previous experiments. </p>

**Evaluating step:**

<p align="justify"> To evaluate the efficacy of our trained models, we employed reads obtained from simulated data and real sequencing experiments that corresponded to the previously trained reference genomes. </p>

<p align="justify"> The first **E. coli dataset** used to test the performance of SVM, RF, LR, NB, and XGBoost is the simulated data set, with a total number of paired-end reads of 200,000. To evaluate the effectiveness of the E. coli-trained model on reads from real sequencing experiments, we utilised a real dataset with accession number SRR625891. </p>

<p align="justify"> The second **S. aureus dataset** used to test the performance of SVM, RF, LR, NB, and XGBoost is the simulated data set with a total number of paired-end reads of 200,000. To evaluate the effectiveness of the S. aureus trained model on reads from real sequencing experiments, we utilised a real dataset from GAUGE for the same genome. </p>

<p align="justify"> The third **Human Chromosome 14** (H. Chr14) dataset used to test the performance of NB is the simulated data set, with a total number of paired-end reads of 300,000. To evaluate the effectiveness of the H. Chr14-trained model on reads from real sequencing experiments, we utilised a real dataset from GAUGE for the same genome. </p>

<p align="justify"> the fourth **Arabidopsis thaliana chromosome 1** dataset used to test the performance of NB is the simulated data set,with a total number of paired-end reads of 100,000. To evaluate the effectiveness of the Arabidopsis thaliana chromosome 1 trained model on real sequencing data, we utilized a real dataset with accession number ERR2173372, targeting the same genome. </p>

<p align="justify"> The fifth **Metriaclima zebra**  dataset used to test the performance of NB is the simulated data set, with a total number of paired-end reads of 300,000.To evaluate the effectiveness of the Metriaclima zebra trained model on real sequencing data, we utilized a real dataset with accession number SRR077289, targeting the same genome. </p>


**The real reads in the sequencing experiments lacked explicit labels, and their accuracy levels were unknown beforehand.**

<p align="justify"> To address this issue in the first dataset (E. coli), we proceeded with aligning the reads to their reference genome through the use of one of the NGS aligners (i.e., BWA). Subsequently, we extracted the computed alignment score for each read. By comparing the predicted labels generated by our trained machine learning models with the labels inferred from the alignment scores, we could effectively evaluate the performance of these models. </p>

<p align="justify"> To address this issue in the second dataset (S. aureus), we assembled the reads generated after testing the data using Velvet assembler. After the assembly, the results are evaluated by one of the assembly evaluation tools called QUAST. We then used Lighter to correct the reads that were classified as errors and assembled the files that contained the correct reads and errors that were corrected by Lighter. 
We are also computing the alignment rates. the alignment statistics computed using Bowtie 2.</p>

To address this issue in the third dataset (H. Chr14), the H. Chr14 reference genome was used using Bowtie 2, and alignment statistics were computed.

To address this issue in the fourth dataset,the Arabidopsis thaliana reference genome was used using Bowtie 2, and alignment statistics were computed.

To address this issue in the fifth dataset ,the  Metriaclima zebra reference genome was used using Bowtie 2, and alignment statistics were computed.
# Tools
We are using different tools in our experiment.

We used the **wgsim** (https://github.com/lh3/wgsim) simulator. Wgsim is a small tool for simulating sequence reads from a reference genome.

<p align="justify"> We also used the **BWA aligner** (https://github.com/lh3/bwa) to align the reads to the E. coli reference genome and extract their alignment scores from the resulting SAM files, and **bowtie2**(https://github.com/BenLangmead/bowtie2) A fast and sensitive gapped read aligner. </p>

We also used **Velvet assembler** (https://github.com/dzerbino/velvet) to assemble the reads that are classified as error-free reads.

We also used **QUAST** (https://github.com/ablab/quast), the assembly evaluation tool, to evaluate the assembly results.

We finally used all of theis:

**Lighter** (https://github.com/mourisl/Lighter), a KMER-based error correction method for genome sequencing data,

**CARE2** (https://github.com/fkallen/CARE) Context-Aware Read Error correction for Illumina reads, 

**Fiona** (https://github.com/seqan/seqan/tree/main/apps/fiona) a parallel and automatic strategy for read error correction,

**RECKONER** (https://github.com/refresh-bio/RECKONER) Read Error Corrector Based on KMC,BFC (https://github.com/lh3/bfc) High-performance error correction for Illumina resequencing data,

**karect** (https://github.com/aminallam/karect) KAUST Assembly Read Error Correction Tool and 

**pollux** (https://github.com/emarinier/pollux) Error correction of second-generation sequencing technologies. we compare all this tools with our model.



# Data


<p align="justify"> The datasets used in this manuscript are the Escherichia coli str. K-12 substr. MG1655 (E. coli) [GenBank: NC_000913.3], and its corresponding real sequencing run is publicly available at NCBI SRA with accession number SRR625891. Also, Staphylococcus aureus (S. aureus) and Human Chromosome 14 (H. Chr14) from GAGE with their corresponding real sequencing reads are publicly available from the GAUGE website (http://gage.cbcb.umd.edu/data/). Also,  Arabidopsis thaliana and its corresponding real sequencing run is publicly available at NCBI SRA with accession number ERR2173372. Also, Metriaclima zebra and its corresponding real sequencing run is publicly available at NCBI SRA with accession number SRR077289. </p>

<p align="justify"> The link to the E. coli and S. aureus simulated datasets used (https://drive.google.com/file/d/10nEAroKXB9uUEFjL8eZ6tle8wr5GryzX/view?usp=drive_link) , the link to the H. Chr14 simulated dataset used (https://drive.google.com/file/d/1XRUYbr7-ytzdYM_kLhkL6c2UZW5h48lV/view?usp=sharing),the link to Arabidopsis thaliana simulated dataset(https://drive.google.com/file/d/1W0wkXSk8LlM7XwIGvG0GWXfQLex9nPmg/view?usp=sharing)  and the link to  Metriaclima zebra simulated dataset(https://drive.google.com/file/d/1XIbIkdnZyJ6LnmtK1Y2fxnvv8Wks3pZu/view?usp=drive_link) to train and test our models </p>


# License
This repository is under MIT license.

# Contact
<p align="justify"> Please do not hesitate to contact us (amira_samy@mans.edu.eg) if you have any comments, suggestions, or clarification requests regarding the study or if you would like to contribute to this resource. </p>





