import numpy as np
import math

def ACE_TDR_Cal(arr_result, rate=0.01):
    ace_list = []
    tdr_list = [0,]
    arr_result = np.array(arr_result)
    total = len(arr_result)
    for thres in arr_result[:,1]:
        TP = TN = FP = FN = 0
        for l,sc in arr_result:
            if sc > thres and l == 1:
                TP = TP + 1.
            elif sc <= thres and l== 0:
                TN = TN + 1.
            elif sc < thres and l==1:
                FN = FN + 1.
            else:
                FP = FP + 1.
        Ferrlive = FP / (FP + TN+1e-7)
        Ferrfake = FN / (FN + TP+1e-7)
        FDR = FP / (FP+TN+1e-7)
        TDR = TP / (TP+FN+1e-7)
        if FDR < rate:
            tdr_list.append(TDR)
        ace_list.append((Ferrlive+Ferrfake)/2.)
    return min(ace_list),max(tdr_list)

def eval_state(probs, labels, thr):
    predict = probs >= thr
    TN = np.sum((labels == 0) & (predict == False))
    FN = np.sum((labels == 1) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    TP = np.sum((labels == 1) & (predict == True))
    return TN, FN, FP, TP

def get_threshold(probs, grid_density):
    Min, Max = min(probs), max(probs)
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(0.0 + i * 1.0 / float(grid_density))
    thresholds.append(1.1)
    return thresholds

def get_EER_states(probs, labels, grid_density = 10000):
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if(FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif(FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list