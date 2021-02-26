### RePReL 

This repository contains an implementation of the paper [Kokel et al. ICAPS 2021]().

> RePReL : Integrating Relational Planning and Reinforcement Learning for Effective Abstraction  
Harsha Kokel, Arjun Manoharan, Sriraam Natarajan, Ravindran Balaraman, Prasad Tadepalli  
In **ICAPS 2021**

### Abstract

State abstraction is necessary for better task transfer in complex reinforcement learning environments. Inspired by the benefit of state abstraction in MAXQ and building upon hybrid planner-RL architectures, we propose RePReL, a hierarchical framework that leverages a relational planner to provide useful state abstractions. Our experiments demonstrate that the abstractions enable faster learning and efficient transfer across tasks. More importantly, our framework enables the application of standard RL approaches for learning in structured domains. The benefit of using the state abstractions is critical in relational settings, where the number and/or types of objects are not fixed apriori. Our experiments clearly show that RePReL framework not only achieves better performance and efficient learning on the task at hand but also demonstrates better generalization to unseen tasks.


### Code organization

```bash
RePReL
├── README.md                       # this file
├── box_world_gym/                  # Gym Environment for Relational Box World domain
├── core/                           # RePReL core code
|  ├── planner/                     # high-level relational planner 
|  ├── reprel_environment/          # RePReL environment specific code
|  |   └──  ...            # contains domain specific abstraction and termination code.
|  └── RePReL_QLearning.py          # RePReL Q-Learning algorithm 
├── data/                           # results from experiments
├── plot/                           # code to visualize results
└── main.py                         # runner file
```

### Citation


Please consider citing the following paper if you find our codes helpful. Thank you!

@inproceedings{kokel2021,
title={{RePReL} : Integrating Relational Planning and Reinforcement Learning for Effective Abstraction},
author={Kokel, Harsha and Manoharan, Arjun and Natarajan, Sriraam and Balaraman, Ravindran and Tadepalli, Prasad  },
booktitle={ICAPS},
year={2021}
}



