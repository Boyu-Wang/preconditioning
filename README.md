# Pre-conditioning strategy for neural networks

This is a course project for CSE 592 Convex Optimization, by Boyu Wang and Yang Wang.

We study the preconditioning matrix for ridge regression. The experiments are performed on a subset of MNIST dataset (digit 4 and 7).

Please check the report for more details. [PDF](report.pdf)


Our proposed preconditioning matrix $$P$$ is:
$$P_k = \Lambda_k ^{-\frac{1}{2}}U_k^T$$,
where $$\Lambda_k$$ and $$U_k$$ is the k-th largest eigen value and vector for correlation matrix $$C=\frac{1}{n}XX^T$$. 

After preconditioning, $$\tilde{X} = P_k X$$. The new average condition number becomes $$k$$.


For Batch optimization, run: 
```
python batch_1fc.py
```

BatchPC.py defines the layer for mini batch Pre-Conditionining.

For mini batch optimization, run:
```
python stochastic_1fc.py
python stochastic_2fc.py
```

For learning preconditioning matrix $$P$$ via regularizer, run:
```
python stochastic_regularize_c.py
```
