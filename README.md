# Unsupervised Joint PoS Tagging and Stemming for Agglutinative Languages

Tested on Windows 10 (x64) with Python 3.5

# Baseline model

**A fully Bayesian approach to unsupervised part-of-speech tagging (Goldwater, Sharon and Griffiths, Tom)**

# Usage

**Baseline Commandline arguments:**
>**input file** - File with sentences one per line 
>Example: Ama hiçbir şey söylemedim ki ben sizlere . Ne ondan bahsedebildim , ne yaşadıklarımdan ... Onlar önemli değil ki .

>**output file** - Output file 
>Example: Ama/9 hiçbir/10 şey/2 söylemedim/3 ki/2 ben/2 sizlere/0 ./10 - word/tag 
>**number of labels**
>**tag size number of iterations**
> **sampling iterations alpha**
> **hyperparameter for transitions beta**
> **hyperparameter for emissions**

# BayesianS-HMM 
**Commandline arguments:** 
>input file - File with sentences one per line Example: Ama hiçbir şey söylemedim ki ben sizlere . Ne ondan bahsedebildim , ne yaşadıklarımdan ... Onlar önemli değil ki .
>output file - Output file (Example: bir&bir/2 tutsağım&tut/6 ben&ben/2 .&./11 - word&stem/tag) 
>number of labels 
>tag size number of iterations
>sampling iterations alpha
>hyperparameter for transitions beta
>hyperparameter for stem emissions

# BayesianSM-HMM 
**Commandline arguments:** 
>input file - File with sentences one per line Example: Ama hiçbir şey söylemedim ki ben sizlere . Ne ondan bahsedebildim , ne yaşadıklarımdan ... Onlar önemli değil ki .
>output file - Output file (Example: vielen&viele+n/9 dank&dan+k/9 !&!+#/10 - word&stem+affix/tag)
>number of labels
>tag size number of iterations
>sampling iterations alpha
>hyperparameter for transitions beta
>hyperparameter for stem emissions gama
>hyperparameter for affix emissions

# BayesianCS-HMM 
**Commandline arguments:** 
input file - File with sentences one per line Example: Ama hiçbir şey söylemedim ki ben sizlere . Ne ondan bahsedebildim , ne yaşadıklarımdan ... Onlar önemli değil ki .
>output file - Output file (Example: bir&bir/2 tutsağım&tut/6 ben&ben/2 .&./11 - word&stem/tag)
>number of labels - tag size number of iterations
>sampling iterations alpha
>hyperparameter for transitions beta
>hyperparameter for stem emissions semantic
>word2vec model

# BayesianCSM-HMM 
**Commandline arguments:** 
>input file - File with sentences one per line Example: Ama hiçbir şey söylemedim ki ben sizlere . Ne ondan bahsedebildim , ne yaşadıklarımdan ... Onlar önemli değil ki .
>output file - Output file (Example: vielen&viele+n/9 dank&dan+k/9 !&!+#/10 - word&stem+affix/tag)
>number of labels
>tag size number of iterations
>sampling iterations alpha
>hyperparameter for transitions beta
>hyperparameter for stem emissions gama
>hyperparameter for affix emissions semantic
>word2vec model

# BayesianA-HMM 
**Commandline arguments:**
>input file - File with sentences one per line Example: Ama hiçbir şey söylemedim ki ben sizlere . Ne ondan bahsedebildim , ne yaşadıklarımdan ... Onlar önemli değil ki .
>output file - Output file (Example: bir&bir/2 tutsağım&tut/6 ben&ben/2 .&./11 - word&stem+affix/tag)
>number of labels
>tag size number of iterations
>sampling iterations alpha
>hyperparameter for transitions beta
>hyperparameter for stem emissions gama
>hyperparameter for affix transitions

# BayesianAS-HMM 
**Commandline arguments:**
>input file - File with sentences one per line Example: Ama hiçbir şey söylemedim ki ben sizlere . Ne ondan bahsedebildim , ne yaşadıklarımdan ... Onlar önemli değil ki .
>output file - Output file (Example: bir&bir/2 tutsağım&tut/6 ben&ben/2 .&./11 - word&stem+affix/tag)
>number of labels
>tag size number of iterations
>sampling iterations alpha
>hyperparameter for transitions beta
>hyperparameter for stem gama
>hyperparameter for affix transitions delta
>hyperparameter for affix emissions

