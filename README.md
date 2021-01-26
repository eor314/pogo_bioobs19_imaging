# pogo_bioobs19_imaging

Authors: Eric Orenstein (Scripps Institution of Oceanography) and Simon-Martin Schroder (University of Kiel)

This set of Jupyter notebooks was designed to introduce biological oceanographers to some tools available for image processing and machine learning. The material was originally presented as part of the <a href="http://www.ocean-partners.org/">Partnership for Observing the Global Ocean</a> <a href="http://ocean-partners.org/pogo-workshop-machine-learning-and-artificial-intelligence-biological-oceanographic-observations">2019 Workshop on Machine Learning and Artifical Intelligence in Biological Observations</a>. 

The notebooks will guide users through manipulating plankton images from the <a href="http://spc.ucsd.edu/">Scripps Plankton Camera System</a> and the <a href="https://sites.google.com/view/piqv/">ZooScan</a>. Labeled data sets for the machine learning examples will be made availabe shortly. Several test images are currently available in the repository for the image processing examples.

Running the notebooks requries a Python 3.6 environment and several particular Python libraries. Appropriate GPU hardware with associated drivers are necessary for accelerated training and execution in the notebooks with deep learning examples. Most of the Python software comes standard with the <a href="https://www.anaconda.com/distribution//">Anaconda</a> Python distribtuion.

# To install dependencies and run Jupyter
<ol>
  <li>Install Anaconda python: https://www.anaconda.com/products/individual</li>
  <li>Install git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git (If installing on Windows check out GitHub Desktop: https://desktop.github.com/)</li>
  <li>Navigate to the local directory where you want the code to live in a terminal</li>
  
        cd PATH/TO/DIRECTORY
  
  <li>Clone into the tutorial’s git repository using the command line:</li>
  
        git clone https://github.com/eor314/pogo_bioobs19_imaging.git
  
  <li>Navigate into the new directory that contains all the project code.</li> 
  <li>Create a new python environment with anaconda. This will allow you to designate the python version and install dependencies without messing with the base installation.</li>
  
        conda create –name pogo-tutorial
  
  <li>Activate the environment</li>
  
        conda activate pogo-tutorial
  
  <li>Start the environment and install a bunch of dependencies</li>

        conda install jupyterlab opencv matplotlib scikit-image scikit-learn pytorch

  <li>Install torchvision</li>
  
        conda install torchvision -c pytorch
        
  <li>Install tqdm (a progress monitoring utility) from conda-forge</li>
  
        conda install -c conda-forge tqdm1
  
  <li>Start jupyter lab via the terminal. It should open a page in your web browser that has an IDE with a file navigation tab open to the directory with all the tutorial code. Be sure to navigate to your POGO tutorial directory before running this command in the future.</li>
  
        jupyter lab
        
  <li>Double click on mod1_image_manipulation. Once it opens, select the box with the import statements and click the play button. If everything installed correctly, it should run with no errors. You’ll know it ran when a number appears in the brackets next to the box.</li>
</ol>

