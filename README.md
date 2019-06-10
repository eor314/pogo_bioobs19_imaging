# pogo_bioobs19_imaging

Authors: Eric Orenstein (Scripps Institution of Oceanography) and Simon-Martin Schroder (University of Kiel)

This set of Jupyter notebooks were designed to introduce biological oceanographers to some tools available for image processing and machine learning. The material was originally presented as part of the <a href="http://www.ocean-partners.org/">Partnership for Observing the Global Ocean</a> <a href="http://ocean-partners.org/pogo-workshop-machine-learning-and-artificial-intelligence-biological-oceanographic-observations">2019 Workshop on Machine Learning and Artifical Intelligence in Biological Observations</a>. 

The notebooks will guide users through manipulating plankton images from the <a href="http://spc.ucsd.edu/">Scripps Plankton Camera System</a> and the <a href="https://sites.google.com/view/piqv/">ZooScan</a>. Labeled data sets for the machine learning examples will be made availabe shortly. Several test images are currently available in the repository for the image processing examples.

Running the notebooks requries a Python 3.6 environment and several particular Python libraries. Appropriate GPU hardware with associated drivers are necessary for accelerated training and execution in the notebooks with deep learning examples. Most of the Python software comes standard with the <a href="https://www.anaconda.com/distribution//">Anaconda</a> Python distribtuion.

Assuming Anaconda is installed, the other necessary dependencies can be loaded using the install_dependencies.sh script. To run it, first navigate to the directory where the repository is installed and make the script executable:

```
chmod +x install_dependencies.sh
```

Then create an Anaconda environment and activate it:

```
conda create -n pogo_workshop pythyon=3
source activate pogo_workshop
```

