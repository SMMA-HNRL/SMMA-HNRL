# SMMA-HNRL
Source code for paper "Prediction of Potential Small Mmolecule−miRNA Associations Based on Heterogeneous Network Representation Learning”.
## Get Started
### Evironment Setting
   Python version >= 3.6
   TensorFlow version >= 1.14
### data
Data obtained from different biological databases are preprocessed and a heterogeneous information network consisting of three association networks (miRNA-SM, miRNA-disease, SM-disease) and three similarity networks (miRNA-miRNA, SM-SM, disease-disease)
### getNodevector
feature vector representations of nodes were obtained by the heterogeneous network representation learning methods HIN2vec and HeGAN
#### Usage
#### Input parameter :
 
      python train.py -m HeGAN -d yourdataset_name
      
      python train.py -m HIN2vec -d yourdataset_name
      
   If you want to train your own dataset, create the file (./dataset/yourdataset_name/edge.txt) and the format is as follows:
       ![image](https://user-images.githubusercontent.com/111487195/185327563-3f3a872d-8cab-49b0-a328-459b68264b06.png)
       
the input graph is directed and the undirected needs to be transformed into directed graph.
       
#### Modle Setup
   The model parameter could be modified in the file ( ./src/config.ini ).
   
   Note: If you want to train your own dataset, you need to declare your own dataset at the beginning of ./src/config.ini
       
#### Common parameter :
      * dim : dimension of output
      
      * epoch : the number of iterations
      
 #### Output :
     
      The results are stored in the file (./output/embedding/HeGAN) and (./output/embedding/HIN2vec).
 ### get SM-miRNA_association_vector
 We use Hadamard, Average, Minus and Absolute Minus to get SM-miRNA pair vector and finally choose Hadamard function to get pair vector
 ### ClassifierSelection
  Predicting SM-miRNA associations could be considered as a binary classification problem. 
  
  We selected five machine learning classification algorithms, NB (Naive Bayes) , LR (Linear Regression), KNN (k-Nearest Neighbor) , AdaBoost and LightGBM.
  ### predictResult
  It contains all the positive samples predicted by the LightGBM classifier
