
'''Indicator Matrices Psedudocode'''
def convert_numbered_targets_to_indicator_matrix(Yin):
    N=len(Yin)
    K=max(Yin)+1
    Yout=zeros(N,K)
    for n in xrange(N):
        Yout[n,Yin[n]]=1
    return Yout
    
 ## Yin values must be from 0,1,2,...K-1   
    
    
prediction_labels=np.argmax(softmax_outputs,axis=1)
target_labels=np.argmax(target_indicator,axis=1)
#prediction_labels=[1,0,2,1]
#target_labels=[1,2,2,0]
#check if correct:
accuracy=sum(prediction_labels ==target_labels)/N

'''argmax()tell us the location of biggest value, it does the inverse of indicator'''