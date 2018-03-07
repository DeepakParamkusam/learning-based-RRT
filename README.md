# learning-based-RRT  
This repository contains the source code of my Master's thesis - "Comparison of optimal control techniques for learning-based RRT".  
 
The thesis can be found [here](https://repository.tudelft.nl/islandora/object/uuid%3A742ed24e-0525-4ae2-b6d4-2dc6f69e60e1).  

The implementations were written in MATLAB, C++ and Python and have the following dependencies :  
• MATLAB Symbolic toolbox  
• [Automatic Control and Dynamic Optimization (ACADO) Toolkit](http://acado.github.io/)  
• numpy Python library  
• [scikit-learn Python library](http://scikit-learn.org/stable/)  
  

The code is divided into different folders in the repository as follows:  
  
**2link_direct** : This folder contains the implementation of direct optimal control for 2-link manipulator. The code is written in C++ and uses the ACADO Toolkit. ACADO toolkit returns solutions in multiple files. They can be consolidated using the scripts in this folder.  
  
**2link_indirect** : Implementation of indirect optimal for 2-link manipulator can be found in this folder. The indirect optimal control equations are generated in MATLAB while the data generation is performed in Python.  
  
**training_data** : Data generated in this thesis is stored here. This folder also contains the Python code used for cleaning the data.  
  
**2link_NN** : The folder contains the code for training the KNN and ANN with the data. scikit-learn library is used for the training. Trained model are stored in trained_models folder.  
  
**2link_rrt** : This folder contains the implementations of learning-based rrt.  

**choosing_rrt** : This folder contains the implementation of choosing-rrt in which control inputs are randomly chosen instead using the optimal control.  


