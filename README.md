# FairCRS: Towards User-oriented Fairness in Conversational Recommendation Systems

# Overview
We propose a framework for mitigating the issues of user unfairness in conversational recommendation systems, i.e., FairCRS. The key idea of the framework is the difference in the performance of CRS models on different user groups due to the modelling bias that we found in the models on different user groups. Specifically, we design a mechanism that enhances the user representation ability of CRS models on disadvantaged user groups - the user-embedding reconstruction mechanism; and we propose a strategy that optimises the difference in recommendation quality between both user groups - the user-oriented fairness strategy. The FairCRS framework not only mitigates user unfairness in existing CRS models but also improves the overall recommendation performance.

![image](https://github.com/LQlq123/FairCRS/blob/main/overallframework.png)
# Environment
* Pip install CRSLab 
* Python == 3.10.9
* Pytorch == 1.13.1

# Dataset
We collected and preprocessed 2 commonly used human annotated datasets (i.e., TG-ReDial and ReDial) and divided each dataset into two user groups (i.e., active and inactive) according to the number of items mentioned by the users in the dialogue, as shown below:

![image](https://github.com/LQlq123/FairCRS/blob/main/dataset.png)

We have uploaded the dataset to [Google Drive](https://drive.google.com/file/d/1a7KutG_JYZnsq0nGcnjY-Ppd4BjzESGv/view?usp=sharing). The downloaded dataset folder should be placed in the "data" folder (needs to be created under the FairCRS path).

# Run
We have applied our FairCRS framework to the CRS baseline, and you can run it directly using the command below:
```bash
python run_crslab.py --config config/crs/kbrd/tgredial.yaml --save_data --save_system
```
This will save the trained model in a folder called "save", and you can run the saved model with the following command:
```bash
python run_crslab.py --config config/crs/kbrd/tgredial.yaml --restore_system
```
Other specific commands can be found in the file "run_crslab.py".

