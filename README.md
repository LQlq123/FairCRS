# FairCRS: Towards User-oriented Fairness in Conversational Recommendation Systems
Please run the code via:

#Overview
We propose Variational Reasoning over Incomplete KGs Conversational Recommender. Our key idea is to incorporate the large dialogue corpus naturally accompanied with CRSs to enhance the incomplete knowledge graphs; and adopt the variational Bayesian method to perform dynamic knowledge reasoning conditioned on the dialogue context. Specifically, we denote the dialogue-specific subgraphs of KGs as latent variables with categorical priors for adaptive knowledge graphs refactor. We propose a variational Bayesian method to approximate posterior distributions over dialogue-specific subgraphs, which not only leverages the dialogue corpus for restructuring missing entity relations but also dynamically selects knowledge based on the dialogue context.

#Environment



#Dataset
我们在


#Run
```bash
python run_crslab.py --config config/crs/kbrd/tgredial.yaml --save_data --save_system
```

#Notation
