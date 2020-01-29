Here, we describe the implementation of the DMRG algorithm to classify images of the mnist dataset. The structure of tasks is as follow:

1. clean.py or preprocess the dataset
2. mps_ini.py initialize the MPS state for the weights of the linear model.
3. left_canon_form.py convert the MPS state into its left canonical form required to perform the feature map.
4. feature_map_update.py perform the updating of the feature map in each running.
5. dmrg.py implement the DMRG algortihm 
