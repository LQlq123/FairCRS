# FairCRS: Towards User-oriented Fairness in Conversational Recommendation Systems
Please run the code via:

# Overview
We propose a framework for mitigating the issues of user unfairness in conversational recommendation systems, i.e., FairCRS. The key idea of the framework is the difference in the performance of CRS models on different user groups due to the modelling bias that we found in the models on different user groups. Specifically, we design a mechanism that enhances the user representation ability of CRS models on disadvantaged user groups - the user-embedded reconstruction mechanism; and we propose a strategy that optimises the difference in recommendation quality between both user groups - the user-oriented fairness strategy. The FairCRS framework not only mitigates user unfairness in existing CRS models but also improves the overall recommendation performance.

![image](https://github.com/LQlq123/FairCRS/blob/main/overallframework.png)
# Environment



# Dataset
我们在


# Run
```bash
python run_crslab.py --config config/crs/kbrd/tgredial.yaml --save_data --save_system
```

# Notation
