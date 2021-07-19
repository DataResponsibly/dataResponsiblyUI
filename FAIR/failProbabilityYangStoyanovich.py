'''
Created on Apr 12, 2017

@author: meike.zehlike
'''
from .yangStoyanovichMethod import create
from .FairnessInRankings import FairnessInRankingsTester

# def determineFailProb(proN,unproN,p,k,alpha=0.01,numTrials=100):
#     """
#     determines the probability that the ranked group fairness test fails given an artificial dataset
#     created by means of Yang and Stoyanovich ("Ke Yang and Julia Stoyanovich. "Measuring Fairness in
#     Ranked Outputs." arXiv preprint arXiv:1610.08559 (2016).") which we believe to be fair.
#
#     """
#     # numTrials = 10000  # Set to 100 or 10,000
#     # alpha = 0.01
#     # k = 1000  # Set to 1000
#
#     result, expectedCandidates = rankedGroupFairnessInYangStoyanovich(alpha, p, k, proN, unproN, numTrials)
#     sumOfFailures = sum(result)
#     failProb = sumOfFailures / numTrials
#     return failProb

#
# def rankedGroupFairnessInYangStoyanovich(alpha, p, k, numberProtected, numberNonProtected, trials):
#     result = [0] * k
#     gft = FairnessInRankingsTester(p, alpha, k, correctedAlpha=False)
#
#     for idx in range(trials):
#         rankedOutput = create(p, k, numberProtected, numberNonProtected)
#         posAtFail, isFair = gft.ranked_group_fairness_condition(rankedOutput)
#         if not isFair:
#             result[posAtFail] += 1
#
#     return result, gft.candidates_needed

def computeFairRankingProbability(k,p,generated_ranking,default_alpha=0.05):
    ## generated_ranking is a list of tuples (id, "pro"),...(id,"unpro")

    gft = FairnessInRankingsTester(p, default_alpha, k, correctedAlpha=True)
    posAtFail, isFair = gft.ranked_group_fairness_condition(generated_ranking)

    if isFair:
        # posAtFail = "NA"
        p_value = gft.calculate_p_value_left_tail(k, generated_ranking)
    else:
        p_value = gft.calculate_p_value_left_tail(posAtFail, generated_ranking)

    return p_value, isFair, posAtFail, gft.alpha_c, gft.candidates_needed

def computeFairPairProbability(small_k,larger_k, p, generated_ranking, default_alpha=0.1):
    ## generated_ranking is a list of tuples (id, "pro"),...(id,"unpro")
    top_k = len(generated_ranking)
    small_k_invalid = (small_k <=0) or (small_k > top_k)
    larger_k_invalid = (larger_k <= 0) or (larger_k > top_k)

    if small_k_invalid:
        p_value_sl = -1
        alpha_c_sl = -1
    else:
        small_gft = FairnessInRankingsTester(p, default_alpha, small_k, correctedAlpha=True)
        p_value_sl = small_gft.calculate_p_value_left_tail(small_k, generated_ranking)
        alpha_c_sl = small_gft.alpha_c

    if larger_k_invalid:
        p_value_lg = -1
        alpha_c_lg = -1
    else:
        larger_gft = FairnessInRankingsTester(p, default_alpha, larger_k, correctedAlpha=True)
        p_value_lg = larger_gft.calculate_p_value_right_tail(larger_k, generated_ranking)
        alpha_c_lg = larger_gft.alpha_c

    return p_value_sl, alpha_c_sl, p_value_lg, alpha_c_lg