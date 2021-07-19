import json
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import linear_model
from math import sqrt
from DataSynthesizer.lib.utils import read_json_file
from FAIR.FairnessInRankings import FairnessInRankingsTester

def save_uploaded_file(file, current_file):
    """
    Save user uploaded data on server.

    Attributes:
        file: the uploaded dataset.
        current_file: file name with out ".csv" suffix
    """
    with open(current_file+".csv", 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

def get_score_scatter(current_file,top_K=100):
    """
    Generated data for scatter plot.

    Attributes:
        current_file: file name that stored the data (with out ".csv" suffix)
        top_K: threshold of data size that scatter plot included
    Return:  data for scatter plot using HighChart format
    """
    data = pd.read_csv(current_file+"_weightsum.csv").head(top_K)
    scatter_points = []
    score_value = data["GeneratedScore"].tolist()
    position_value = [x for x in range(1, len(data) + 1)]
    for i in range(len(score_value)):
        scatter_points.append([position_value[i], score_value[i]])
    return scatter_points

def getAttValueCountTopAndOverall(input_data, att_name, top_K=10):
    """
            Subfunction to count values of input attribute in the data for top 10 and overall pie chart.

            Attributes:
                input_data: dataframe that store the input data
                att_name: name of attribuet to count
                top_K: top k position to count the value, default value is 10
            Return:  json data includes two two-dimension arrays for value and its count at top 10 and overall
            """
    counts_all = {}
    all_values_count = input_data[att_name].value_counts()
    top_data = input_data[0:top_K]
    # get overall counts
    new_values_all = []
    for i in range(len(all_values_count)):
        cur_cate = all_values_count.index[i]
        # if not a string, then encode it to the type that is JSON serializable
        if not isinstance(cur_cate, str):
            cur_cate = str(cur_cate)
        cur_count = int(all_values_count.values[i])
        new_values_all.append([cur_cate,cur_count])
    counts_all["overall"] = new_values_all

    # get top K counts and make sure list of counts include every value of input attribute for consistent pie chart colors
    top_values_count = top_data[att_name].value_counts()
    top_cates = top_values_count.index
    # generate a dict to store the top k value counts
    top_values_count_dic = {}
    for i in range(len(top_values_count)):
        top_values_count_dic[top_values_count.index[i]] = int(top_values_count.values[i])
    # generate a new value list for top K using same order as in over all list
    new_values_top = []
    for i in range(len(all_values_count)):
        cur_cate = all_values_count.index[i]
        # if not a string, then encode it to the type that is JSON serializable
        if not isinstance(cur_cate, str):
            str_cur_cate = str(cur_cate)
        else:
            str_cur_cate = cur_cate
        if cur_cate in top_cates: # exiting in top K
            new_values_top.append([str_cur_cate, top_values_count_dic[cur_cate]])
        else:
            new_values_top.append([str_cur_cate, 0])
    counts_all["topTen"] = new_values_top
    return counts_all


def get_chart_data(current_file, att_names):
    """
        Generated data for pie chart.

        Attributes:
            current_file: file name that stored the data (with out ".csv" suffix)
            att_names: list of attribute names to compute the chart data
        Return:  json data for pie chart plot using HighChart format
        """
    data = pd.read_csv(current_file + "_weightsum.csv")
    pie_data = {}
    for ai in att_names:
        cur_ai_json = {}
        counts_all = getAttValueCountTopAndOverall(data,ai)
        cur_ai_json["topTen"] = counts_all["topTen"]
        cur_ai_json["overall"] = counts_all["overall"]
        pie_data[ai] = cur_ai_json
    return pie_data


def computeSlopeOfScores(current_file,top_K, round_default=2):
    """
    Compute the slop of scatter plot.

    Attributes:
        current_file: file name that stored the data (with out ".csv" suffix)
        top_K: threshold of data size to compute the slope
    Return:  slope of scatter plot of top_K data
    """
    data = pd.read_csv(current_file + "_weightsum.csv")
    top_data = data[0:top_K]
    xd = [i for i in range(1,top_K+1)]
    yd = top_data["GeneratedScore"].tolist()
    # determine best fit line
    par = np.polyfit(xd, yd, 1, full=True)
    slope = par[0][0]
    return round(slope,round_default)

def compute_correlation(current_file,y_col="GeneratedScore",top_threshold=3,round_default=2):
    """
    Compute the correlation between attributes and generated scores.

    Attributes:
        current_file: file name that stored the data (with out ".csv" suffix)
        y_col: column name of Y variable
        top_threshold: threshold of number of returned correlated attribute
        round_default: threshold of round function for the returned coefficient
    Return:  list of correlated attributes and its coefficients
    """
    # get the data for generated ranking
    ranking_df = pd.read_csv(current_file+"_weightsum.csv")
    # get the upload data for correlation computation
    upload_df = pd.read_csv(current_file+".csv")

    numeric_atts = list(upload_df.describe().columns)
    X = upload_df[numeric_atts].values
    #  no need to standardize data
    # scaler = StandardScaler()
    # transform_X = scaler.fit_transform(X)
    y = ranking_df[y_col].values

    regr = linear_model.LinearRegression(normalize=False)
    regr.fit(X, y)

    # get coeff's, ordered by significance
    # format weight with decile of 3
    for i in range(len(regr.coef_)):
        regr.coef_[i] = round(regr.coef_[i], round_default)
    # normalize coefficients to [-1,1]
    max_coef = max(regr.coef_)
    min_coef = min(regr.coef_)
    abs_max = max(abs(max_coef),abs(min_coef))
    stand_coef = []
    for ci in regr.coef_:
        new_ci = round(ci/abs_max,round_default)
        stand_coef.append(new_ci)

    # coeff_zip = zip(regr.coef_, numeric_atts)
    coeff_zip = zip(stand_coef, numeric_atts)
    coeff_sorted = sorted(coeff_zip, key=lambda tup: abs(tup[0]), reverse=True)

    if len(coeff_sorted) > top_threshold:
        coeff_return = coeff_sorted[0:top_threshold]
    else:
        coeff_return = coeff_sorted

    # only return top_threshold most correlated attributes
    return coeff_return

def compute_statistic_topN(chosed_atts,current_file,top_N,round_default=1):
    """
    Compute the statistics of input attributes.

    Attributes:
        chosed_atts: list of attributes to be computed
        current_file: file name that stored the data (with out ".csv" suffix)
        top_N: size of data to be used in current_file
        round_default: threshold of round function for the returned statistics
    Return:  json data of computed statistics
    """
    # data is sorted by ranking scores from higher to lower
    data = pd.read_csv(current_file+"_weightsum.csv").head(top_N)
    statistic_data = {}
    # get the median data
    for atti in chosed_atts:
        cur_att_max = max(data[atti])
        cur_att_median = np.median(data[atti])
        cur_att_min = min(data[atti])
        statistic_data[atti] = {"max": round(cur_att_max, round_default),
                        "median": round(cur_att_median, round_default),
                        "min": round(cur_att_min, round_default)}
    return statistic_data



def mergeUnfairRanking(_px, _sensitive_idx, _fprob):  # input is the ranking
    """
    Generate a fair ranking.

    Attributes:
        _px: input ranking (sorted), list of ids
        _sensitive_idx: the index of protected group in the input ranking
        _fprob: probability to choose the protected group
    Return:  generated fair ranking, list of ids
    """
    #     _px=sorted(range(len(_inputrankingscore)), key=lambda k: _inputrankingscore[k],reverse=True)
    rx = [x for x in _px if x not in _sensitive_idx]
    qx = [x for x in _px if x in _sensitive_idx]
    rx.reverse()  # prepare for pop function to get the first element
    qx.reverse()
    res_list = []

    while (len(qx) > 0 and len(rx) > 0):
        r_cur = random.random()
        #         r_cur=random.uniform(0,1.1)
        if r_cur < _fprob:
            res_list.append(qx.pop())  # insert protected group first
        else:
            res_list.append(rx.pop())

    if len(qx) > 0:
        qx.reverse()
        res_list = res_list + qx
    if len(rx) > 0:
        rx.reverse()
        res_list = res_list + rx

    if len(res_list) < len(_px):
        print("Error!")
    return res_list

def runFairOracles(chosed_atts,current_file,alpha_default=0.05,k_threshold=200,k_percentage=0.5):
    """
    Run all fairness oracles: FA*IR, Pairwise and Proportion

    Attributes:
        chosed_atts: list of sensitive attributes
        current_file: file name that stored the data (with out ".csv" suffix)
        alpha_default: default value of significance level in each oracle
        k_threshold: threshold of size of upload data to decide the top-K in FA*IR and Proportion
        k_percentage: threshold to help to decide the top-K in FA*IR and Proportion when upload dataset's size less than k_threshold
    Return:  json data of fairness results of all oracles
    """
    # data is sorted by ranking scores from higher to lower
    data = pd.read_csv(current_file+"_weightsum.csv")
    total_n = len(data)
    # set top K based on the size of input data
    # if N > 200, then set top K = 100, else set top K = 0.5*N
    if total_n > k_threshold:
        top_K = 100
    else:
        top_K = int(np.ceil(k_percentage* total_n))

    fair_res_data = {} # include all details of fairness validation
    fair_statement_data = {} # only include the fairness result, i.e. fair or unfair, True represents fair, False represents unfair.
    for si in chosed_atts:
        # get the unique value of this sensitive attribute
        values_si_att = list(data[si].unique())
        # for each value, compute the current pairs and estimated fair pairs
        si_value_json = {}
        si_fair_json = {}
        for vi in values_si_att:
            # run FAIR oracle to compute its p-value and alpha_c
            p_value_fair,alphac_fair = computePvalueFAIR(si,vi,current_file,top_K)
            # res_fair= p_value_fair > alphac_fair

            # for extreme case, i.e. very few protected group
            if alphac_fair == 0:
                res_fair = "NA"
            else:
                if p_value_fair > alphac_fair:
                    res_fair = "fair"
                else:
                    res_fair = "unfair"

            # run Pairwise orace to compute its p-value, alpha use the default value
            if current_file[0:2] not in ["GC", "PP", "CS"]: # run simulation
                p_value_pairwise = computePvaluePairwise_simu(si, vi, current_file)
            else: # use exampled data set pairwise results
                p_value_pairwise = computePvaluePairwise(si,vi,current_file)
            # res_pairwise = p_value_pairwise > alpha_default
            # for extreme case, i.e. very few protected group
            if p_value_pairwise == 0:
                res_pairwise = "NA"
            else:
                if p_value_pairwise > alpha_default:
                    res_pairwise = "fair"
                else:
                    res_pairwise = "unfair"

            # run Proportion oracle to compute its p-value, alpha use the default value
            p_value_proportion = computePvalueProportion(si,vi,current_file,top_K)
            # res_proportion = p_value_proportion > alpha_default
            if p_value_proportion > alpha_default:
                res_proportion = "fair"
            else:
                res_proportion = "unfair"

            if not isinstance(vi, str):
                filled_vi = vi
            else:
                filled_vi = vi.replace(" ", "")

            si_value_json[filled_vi] = [p_value_fair,alphac_fair,p_value_pairwise,alpha_default,p_value_proportion,alpha_default]
            si_fair_json[filled_vi] = [res_fair,res_pairwise,res_proportion]

        if not isinstance(si, str):
            filled_si = si
        else:
            filled_si = si.replace(" ", "")
        fair_res_data[filled_si] = si_value_json
        fair_statement_data[filled_si] = si_fair_json
    return fair_res_data, fair_statement_data, alpha_default, top_K

def computePvalueFAIR(att_name,att_value,current_file,top_K,round_default=2):
    """
    Compute p-value using FA*IR oracle

    Attributes:
        att_name: sensitive attribute name
        att_value: value of protected group of above attribute
        current_file: file name that stored the data (with out ".csv" suffix)
        top_K: top_K value in FA*IR
        round_default: threshold of round function for the returned p-value
    Return:  rounded p-value and adjusted significance level in FA*IR
    """
    # input checked_atts includes names of checked sensitive attributes
    data = pd.read_csv(current_file + "_weightsum.csv")
    total_N = len(data)
    top_data = data[0:top_K]

    # for attribute value, compute the current pairs and estimated fair pairs
    position_lists_val = data[data[att_name]==att_value].index+1
    size_vi = len(position_lists_val)

    fair_p_vi = size_vi/total_N

    # generate a ranking of tuples with (id,"pro")/(id,"unpro") by current value as protected group
    generated_ranking = []
    for index, row in top_data.iterrows():
        if row[att_name] == att_value:
            generated_ranking.append([index,"pro"])
        else:
            generated_ranking.append([index,"unpro"])

    p_value, isFair, posiFail, alpha_c, pro_needed_list = computeFairRankingProbability(top_K,fair_p_vi,generated_ranking)
    return round(p_value,round_default),round(alpha_c,round_default)

def computePvaluePairwise(att_name,att_value,current_file, round_default=2):
    """
    Compute p-value using Pairwise oracle

    Attributes:
        att_name: sensitive attribute name
        att_value: value of protected group of above attribute
        current_file: file name that stored the data (with out ".csv" suffix)
        run_time: running times of simulation using mergeUnfairRanking
        round_default: threshold of round function for the returned p-value
    Return:  rounded p-value
    """
    data = pd.read_csv(current_file + "_weightsum.csv")
    total_N = len(data)

    # for attribute value, compute the current pairs and estimated fair pairs
    position_lists_val = data[data[att_name] == att_value].index + 1
    size_vi = len(position_lists_val)

    fair_p_vi = size_vi / total_N

    # get the pre-computed pairwise results from simulation
    simu_data = read_json_file("./playdata/rankingfacts/SimulationPairs_N" + str(total_N) + "_R1000.json")

    all_fair_p = list(simu_data.keys())
    if str(fair_p_vi) in all_fair_p:
        cur_pi = str(fair_p_vi)
    else:
        diff_p = []
        for pi in all_fair_p:
            num_pi = float(pi)
            diff_p.append(abs(num_pi - fair_p_vi))

        min_diff_index = diff_p.index(min(diff_p))
        cur_pi = all_fair_p[min_diff_index]
    # compute the number of pairs of value > * in the input ranking that is stored in the current file
    pair_N_vi, estimated_fair_pair_vi, size_vi = computePairN(att_name,att_value,current_file)

    # compute the cdf, i.e. p-value of input pair value
    sample_pairs = simu_data[cur_pi]

    cdf_pair = Cdf(sample_pairs,pair_N_vi)
    # decide to use left tail or right tail
    # mode_pair_sim,_ = mode(sample_pairs)
    # median_mode = np.median(list(mode_pair_sim))
    # if pair_N_vi <= mode_pair_sim:
    #     p_value = cdf_pair
    # else:
    #     p_value = 1- cdf_pair
    return round(cdf_pair,round_default)

def computePvaluePairwise_simu(att_name,att_value,current_file, run_time=100, round_default=2):
    """
    Compute p-value using Pairwise oracle

    Attributes:
        att_name: sensitive attribute name
        att_value: value of protected group of above attribute
        current_file: file name that stored the data (with out ".csv" suffix)
        run_time: running times of simulation using mergeUnfairRanking
        round_default: threshold of round function for the returned p-value
    Return:  rounded p-value
    """
    data = pd.read_csv(current_file + "_weightsum.csv")
    total_N = len(data)

    # for attribute value, compute the current pairs and estimated fair pairs
    position_lists_val = data[data[att_name] == att_value].index + 1
    size_vi = len(position_lists_val)

    fair_p_vi = size_vi / total_N

    seed_random_ranking = [x for x in range(total_N)]  # list of IDs
    seed_f_index = [x for x in range(size_vi)]  # list of IDs

    # for simulation outputs
    data_file = "./media/FairRankingGeneration"
    plot_df = pd.DataFrame(columns=["RunCount", "N", "sensi_n", "fair_mp", "pair_n"])
    # run simulations, in each simulation, generate a fair ranking with input N and size of sensitive group
    for ri in range(run_time):
        # only for binary sensitive attribute
        output_ranking = mergeUnfairRanking(seed_random_ranking, seed_f_index, fair_p_vi)
        position_pro_list = [i for i in range(len(output_ranking)) if output_ranking[i] in seed_f_index]
        count_sensi_prefered_pairs = 0
        for i in range(len(position_pro_list)):
            cur_position = position_pro_list[i]
            left_sensi = size_vi - (i + 1)
            count_sensi_prefered_pairs = count_sensi_prefered_pairs + (total_N - cur_position - left_sensi)
        # count_other_prefered_pairs = (_input_sensi_n*(_input_n-_input_sensi_n)) - count_sensi_prefered_pairs
        cur_row = [ri + 1, total_N, size_vi, fair_p_vi, count_sensi_prefered_pairs]
        plot_df.loc[len(plot_df)] = cur_row
    # save the data of pairs in fair ranking generation on server
    plot_df.to_csv(data_file + "_R" + str(run_time) + "_N" + str(total_N) + "_S" + str(size_vi) + "_pairs.csv")
    # compute the number of pairs of value > * in the input ranking that is stored in the current file
    pair_N_vi, estimated_fair_pair_vi, size_vi = computePairN(att_name,att_value,current_file)

    # compute the cdf, i.e. p-value of input pair value
    sample_pairs = list(plot_df["pair_n"].dropna())

    cdf_pair = Cdf(sample_pairs,pair_N_vi)
    # decide to use left tail or right tail
    # mode_pair_sim,_ = mode(sample_pairs)
    # median_mode = np.median(list(mode_pair_sim))
    # if pair_N_vi <= mode_pair_sim:
    #     p_value = cdf_pair
    # else:
    #     p_value = 1- cdf_pair
    return round(cdf_pair,round_default)

def computePvalueProportion(att_name,att_value,current_file, top_K, round_default=2):
    """
    Compute p-value using Proportion oracle, i.e., z-test method of 4.1.3 in "A survey on measuring indirect discrimination in machine learning".

    Attributes:
        att_name: sensitive attribute name
        att_value: value of protected group of above attribute
        current_file: file name that stored the data (with out ".csv" suffix)
        top_K: threshold to decide the positive outcome. Ranked inside top_K is positive outcome. Otherwise is negative outcome.
        round_default: threshold of round function for the returned p-value
    Return:  rounded p-value
    """
    # using z-test method of 4.1.3 in "A survey on measuring indirect discrimination in machine learning"
    # for binary attribute only
    data = pd.read_csv(current_file + "_weightsum.csv")
    total_N = len(data)
    top_data = data[0:top_K]
    # for attribute value, compute the current pairs and estimated fair pairs
    position_lists_val = data[data[att_name] == att_value].index + 1
    size_vi = len(position_lists_val)
    size_other = total_N - size_vi

    size_vi_top = len(top_data[top_data[att_name]==att_value].index +1)
    size_other_top = top_K - size_vi_top

    p_vi_top = size_vi_top / size_vi
    p_other_top = size_other_top / size_other

    p_vi_rest = 1 - p_vi_top
    p_other_rest = 1- p_other_top

    pooledSE = sqrt((p_vi_top * p_vi_rest/ size_vi) + ( p_other_top * p_other_rest / size_other))
    z_test = (p_other_top - p_vi_top) / pooledSE
    p_value = norm.sf(z_test)

    return round(p_value,round_default)


def Cdf(_input_array, x):
    """
    Compute the CDF value of input samples using left tail computation
    Attributes:
        _input_array: list of data points
        x: current K value
    Return:  value of cdf
    """
    # left tail
    count = 0.0
    for vi in _input_array:
        if vi <= x:
            count += 1.0
    prob = count / len(_input_array)
    return prob

def computeFairRankingProbability(k,p,generated_ranking,default_alpha=0.05):
    """
    Sub-function to compute p-value used in FA*IR oracle

    Attributes:
        k: top_K value in FA*IR
        p: minimum proportion of protected group
        generated_ranking: input ranking of users
        default_alpha: default significance level of FA*IR
    Return:  p-value, fairness, rank position fail, adjusted significance level and list of ranking positions that protected group should be using FA*IR
    """
    ## generated_ranking is a list of tuples (id, "pro"),...(id,"unpro")

    gft = FairnessInRankingsTester(p, default_alpha, k, correctedAlpha=True)
    posAtFail, isFair = gft.ranked_group_fairness_condition(generated_ranking)

    p_value = gft.calculate_p_value_left_tail(k, generated_ranking)

    return p_value, isFair, posAtFail, gft.alpha_c, gft.candidates_needed


def computePairN(att_name, att_value,current_file):
    """
    Sub-function to compute number of pairs that input value > * used in Pairwise oracle

    Attributes:
        att_name: sensitive attribute name
        att_value: value of protected group of above attribute
        current_file: file name that stored the data (with out ".csv" suffix)
    Return:  number of pairs of att_value > * in input data, number of pairs of att_value > * estimated using proportion, and proportion of group with att_value
    """
    # input checked_atts includes names of checked sensitive attributes
    data = pd.read_csv(current_file + "_weightsum.csv")
    total_N = len(data)
    # get the unique value of this sensitive attribute
    values_att = list (data[att_name].unique())
    # for each value, compute the current pairs and estimated fair pairs

    position_lists_val = data[data[att_name]==att_value].index+1
    size_vi = len(position_lists_val)
    count_vi_prefered_pairs = 0
    for i in range(len(position_lists_val)):
        cur_position = position_lists_val[i]
        left_vi = size_vi - (i + 1)
        count_vi_prefered_pairs = count_vi_prefered_pairs + (total_N - cur_position - left_vi)
    # compute estimated fair pairs
    total_pairs_vi = size_vi*(total_N-size_vi)
    estimated_vi_pair = math.ceil((size_vi / total_N) * total_pairs_vi)

    return int(count_vi_prefered_pairs),int(estimated_vi_pair),int(size_vi)


def getSizeOfRanking(current_file):
    """
    Compute size of generated ranking.

    Attributes:
        current_file: file name that stored the data (with out ".csv" suffix)
    Return:  size of ranking
    """
    data = pd.read_csv(current_file+"_weightsum.csv")
    return len(data)

def getSizeOfDataset(current_file):
    """
    Compute number of rows in the input data.

    Attributes:
        current_file: file name that stored the data (with out ".csv" suffix)
    Return:  number of rows in current_file
    """
    data = pd.read_csv(current_file+".csv")
    return len(data)

def generateRanking(current_file,top_K=100):
    """
    Generate a ranking of input data.

    Attributes:
        current_file: file name that stored the data (with out ".csv" suffix)
        top_K: threshold of returned generated ranking
    Return:  json data of a dataframe that stored the generated ranking
    """
    ranks_file = current_file + "_rankings.json"
    rankings_paras = read_json_file(ranks_file)
    data = pd.read_csv(current_file + ".csv")
    # before compute the score, replace the NA in the data with 0
    filled_data = data.fillna(value=0)
    chosed_atts = rankings_paras["ranked_atts"]
    filled_data["GeneratedScore"] = 0
    for i in range(len(chosed_atts)):
        cur_weight = rankings_paras["ranked_atts_weight"][i]
        filled_data["GeneratedScore"] += cur_weight * filled_data[chosed_atts[i]]
    filled_data = filled_data.reindex_axis(['GeneratedScore'] + list([a for a in filled_data.columns if a != 'GeneratedScore']), axis=1)
    # save data with weight sum to a csv on server
    filled_data.sort_values(by="GeneratedScore",ascending=False,inplace=True)
    filled_data.to_csv(current_file+"_weightsum.csv", index=False)
    # only show top_K rows in the UI
    display_data = filled_data.head(top_K)
    return display_data.to_json(orient='records')



def standardizeData(inputdata,colums_to_exclude=[]):
    """
        inputdata is a dataframe stored all the data read from a csv source file
        noweightlist is a array like data structure stored the attributes which should be ignored in the normalization process.
        return the distribution of every attribute
    """
    df = inputdata.loc[:, inputdata.columns.difference(colums_to_exclude)]# remove no weight attributes
    df_stand = (df - df.mean())/np.std(df)
    inputdata.loc[:, inputdata.columns.difference(colums_to_exclude)] = df_stand
    return inputdata

def normalizeDataset(input_file_name,noweightlist=[]):
    """
        inputdata is the file name of the csv source file
        noweightlist is a array like data structure stored the attributes which should be ignored in the normalization process.
        return the processed inputdata
    """
    input_data = pd.read_csv(input_file_name)
    df = input_data.loc[:,input_data.columns.difference(noweightlist)] # remove no weight attributes
    #normalize attributes
    norm_df = (df - df.min()) / (df.max() - df.min())

    input_data.loc[:,input_data.columns.difference(noweightlist)] = norm_df

    return input_data

def cleanseData(input_file_name, columns_to_exclude=[]):
    """
            inputdata is the file name of the csv source file
            noweightlist is a array like data structure stored the attributes which should be ignored in the normalization process.
            return the cleansed inputdata using normalizating and standization
        """
    norm_data = normalizeDataset(input_file_name, columns_to_exclude)
    return standardizeData(norm_data, columns_to_exclude)

class DataDescriberUI(object):
    """Analyze input dataset, then save the dataset description in a JSON file.
       Used to display in datatable.
    Attributes:
        threshold_size: float, threshold when size of input_dataset exceed this value, then only display first 100 row in input_dataset
        dataset_description: Dict, a nested dictionary (equivalent to JSON) recording the mined dataset information.

        input_dataset: the dataset to be analyzed.

    """

    def __init__(self, threshold_size=100):
        self.threshold_size = threshold_size
        self.dataset_description = {}
        self.input_dataset = pd.DataFrame()
        self.json_data = {}

    def read_dataset_from_csv(self, file_name=None):
        try:
            self.input_dataset = pd.read_csv(file_name)
        except (UnicodeDecodeError, NameError):
            self.input_dataset = pd.read_csv(file_name, encoding='latin1')

        num_tuples, num_attributes = self.input_dataset.shape
        if num_tuples > self.threshold_size:
            self.display_dataset = self.input_dataset.head(100)
        else:
            self.display_dataset = self.input_dataset

    def get_dataset_meta_info(self):
        num_tuples, num_attributes = self.input_dataset.shape
        attribute_list = self.input_dataset.columns.tolist()

        meta_info = {"num_tuples": num_tuples, "num_attributes": num_attributes, "attribute_list": attribute_list}
        self.dataset_description['meta'] = meta_info

    def get_json_data(self):
        self.json_data = self.display_dataset.to_json(orient='records')

    def save_dataset_description_to_file(self, file_name):
        with open(file_name, 'w') as outfile:
            json.dump(self.dataset_description, outfile, indent=4)

    def save_dataset_to_file(self, file_name):

        with open(file_name, 'w') as outfile:
            outfile.write(str(self.json_data))

    def display_dataset_description(self):
        print(json.dumps(self.dataset_description, indent=4))
