import numpy as np
a=np.random.randn(5)
expa=np.exp(a)

answer= expa/expa.sum()



A=np.random.randn(100,5)
expA=np.exp(A)
answer=expA/expA.sum(axis=1,keepdims=True)#sum along columns

answer.sum(axis=1,keepdims=True).shape      #(100L, 1L)
expA.sum(axis=1,keepdims=True).shape #(100L, 1L)
