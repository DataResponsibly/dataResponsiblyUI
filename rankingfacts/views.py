import json
import pandas as pd
import numpy as np
from time import time

from django.urls import reverse
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render

from synthesizer.lib.DataSynthesizerWrapper import get_histograms_of
from synthesizer.lib.DataSynthesizerWrapper import get_categorical_attributes_csv
from synthesizer.lib.DataSynthesizerWrapper import get_binary_attributes_csv
from .models import DataDescriberUI
from .models import get_score_scatter
from .models import save_uploaded_file
from .models import generateRanking
from .models import compute_statistic_topN
from .models import cleanseData

from .models import computeSlopeOfScores
from .models import runFairOracles

from .models import getSizeOfRanking
from .models import getSizeOfDataset
from .models import compute_correlation
from .models import get_chart_data
from DataSynthesizer.lib.utils import read_json_file


# view functions to handle the landing page
def base(request):
    # assign each user a specific Id using the time stamp
    cur_time_stamp = str(int(time()*1e7))
    cur_user_id = "U"+cur_time_stamp
    # context = {"passed_user_id": cur_user_id}
    request.session['passed_user_id'] = cur_user_id
    return render(request, "rankingfacts/webpage.html")

def upload_data(request):
    cur_user_id = request.session.get('passed_user_id')
    # NOTICE: no .csv suffix in current passed file name
    data_server_path = "./media/"

    if request.POST:
        if request.FILES:
            # get user upload file
            upload_csvfile = request.FILES['user_upload_data']
            current_data_name = data_server_path + cur_user_id
            save_uploaded_file(upload_csvfile, current_data_name)
            request.session['passed_data_name'] = current_data_name
        return HttpResponseRedirect(reverse('rankingfacts:data_process'))
    else:
        return render(request, "rankingfacts/upload_data_boot.html")

def sample_data(request):
    cur_user_id = request.session.get('passed_user_id')
    # NOTICE: no .csv suffix in current passed file name
    server_data_names_map = {"GermanCredit_age25": "GC", "ProPublica_gender": "PP", "CSranking_faculty30": "CS"}
    # create a time stamp for current uploading
    data_server_path = "./media/"
    play_data_server_path = "./playdata/rankingfacts/"
    if request.POST:
        selected_data = request.POST["dataset_select"]
        # create a copy of current play data set on server to allow differentiate multiple users at the same time
        cur_data = pd.read_csv(play_data_server_path + selected_data + ".csv")
        new_stamped_name = data_server_path + server_data_names_map[selected_data] + cur_user_id
        cur_data.to_csv(new_stamped_name + ".csv", index=False)
        request.session['passed_data_name'] = new_stamped_name
        return HttpResponseRedirect(reverse('rankingfacts:data_process'))
    else:
        return render(request, "rankingfacts/sample_data_boot.html")

# view functions to handle unprocessing parameters setings part
def data_process(request):
    # NOTICE: no .csv suffix in current passed file name
    passed_data_name = request.session.get('passed_data_name')

    atts_info = get_histograms_of(passed_data_name + ".csv")
    # get the categorical attribuet from original uploaded dataset
    cat_atts = atts_info["cate_atts"]
    cat_atts_ids = ["att"+str(i) for i in range(len(cat_atts))]
    zip_cate_atts = zip(cat_atts, cat_atts_ids)

    att_list = atts_info["all_atts"]
    att_ids_list = ["att"+str(i) for i in range(len(att_list))]
    num_atts_names = atts_info["numeric_atts"]
    num_atts_ids = ["att" + str(i) for i in range(len(num_atts_names))]
    zip_num_atts = zip(num_atts_names, num_atts_ids)

    # get the binary attribute from original uploaded dataset
    binary_atts = get_binary_attributes_csv(passed_data_name + ".csv")
    binary_att_ids = ["binary_att" + str(i) for i in range(len(binary_atts))]
    zip_binarys = zip(binary_atts, binary_att_ids)

    row_binary_atts = []
    for bai in range(len(binary_att_ids)):
        if bai % 3 ==0:
            row_binary_atts.append("endrow")
        else:
            row_binary_atts.append("row")
    row_cat_atts = []
    for cai in range(len(cat_atts_ids)):
        if cai % 3 ==0:
            row_cat_atts.append("endrow")
        else:
            row_cat_atts.append("row")
    row_num_atts = []
    for ai in range(len(num_atts_ids)):
        if ai % 3 == 0:
            row_num_atts.append("endrow")
        else:
            row_num_atts.append("row")
    num_binary_att = zip(binary_atts, binary_att_ids, row_binary_atts)
    num_cat_att = zip(cat_atts, cat_atts_ids, row_cat_atts)
    num_all_att = zip(num_atts_names, num_atts_ids, row_num_atts)
    # initialize json data and header for dataTables
    json_data_table = []
    json_header_table = []
    for i in range(len(att_list)):
        json_data_table.append({"data": str(att_list[i])})
        json_header_table.append({"title": str(att_list[i]),"targets": i})

    if request.POST:
        # if user submit, save user input to server as parameter file
        # get user input
        json_parameter = {}
        included_atts = []
        included_atts_ids = []
        atts_weights = []

        for i in range(len(num_atts_names)):
            atti_id = num_atts_ids[i]
            atti_name = num_atts_names[i]

            atti_json = {}
            att_weight = request.POST[atti_id + "_weight"]
            att_rank = request.POST[atti_id + "_checks"]
            att_order = request.POST[atti_id + "_order"]
            if att_weight =="": # if empty then use default value 1.0
                att_weight = 1.0
            atti_json["weight"] = att_weight
            atti_json["rank"] = att_rank
            atti_json["order"] = att_order

            if att_rank == atti_name:
                ranked_weight = float(att_weight)
                included_atts.append(atti_name)
                included_atts_ids.append(atti_id)
                if att_order == "lower":
                    ranked_weight = - ranked_weight
                atts_weights.append(ranked_weight)

            json_parameter[atti_id] = atti_json
        json_parameter["ranked_atts"] = included_atts
        json_parameter["ranked_atts_weight"] = atts_weights
        # get checked binary sensitive attribute
        checked_sensi_atts = []
        checked_sensi_att_ids = []
        for vi in range(len(binary_atts)):
            vi_name = binary_atts[vi]
            vi_id = binary_att_ids[vi]
            att_sensi = request.POST[vi_id + "_sensi_checks"]
            if att_sensi == vi_name:
                checked_sensi_atts.append(vi_name)
                checked_sensi_att_ids.append(vi_id)
        json_parameter["checked_sensi_atts"] = checked_sensi_atts

        # get checked categorical sensitive attribute
        checked_cate_atts = []
        checked_cate_att_ids = []
        for vi in range(len(cat_atts)):
            vi_name = cat_atts[vi]
            vi_id = cat_atts_ids[vi]
            att_cate = request.POST[vi_id + "_cate_checks"]
            if att_cate == vi_name:
                checked_cate_atts.append(vi_name)
                checked_cate_att_ids.append(vi_id)
        json_parameter["checked_cate_atts"] = checked_cate_atts

        # save user input parameters to server
        json_parameter_outputfn = passed_data_name + "_rankings.json"
        with open(json_parameter_outputfn, 'w') as outfile:
            json.dump(json_parameter, outfile, indent=4)

        res_ranked_cols = ["GeneratedScore"]
        res_ranked_cols = res_ranked_cols + included_atts
        json_rank_table = []
        json_rank_header_table = []
        for i in range(len(res_ranked_cols)):
            json_rank_table.append({"data": str(res_ranked_cols[i])})
            json_rank_header_table.append({"title": str(res_ranked_cols[i]),"targets": i})

        context = {'passed_data_name': passed_data_name, "passed_json_columns": json_data_table,
                   "passed_column_name": att_list, "drawable_atts": atts_info["drawable_atts"],
                   "passed_cate_atts": cat_atts, "passed_atts_json": json_parameter,
                   "passed_ranked_atts": included_atts, "passed_res_atts": res_ranked_cols,
                   "passed_json_ranks": json_rank_table, "passed_json_columns_header": json_header_table,
                   "passed_cols_ids": att_ids_list,"passed_json_ranks_header":json_rank_header_table,
                   "passed_ranked_atts_ids": included_atts_ids,"passed_zip_numeric":zip_num_atts,
                   "passed_num_atts_ids": num_atts_ids,"passed_num_atts_names":num_atts_names,
                   "passed_binary_zips": zip_binarys, "passed_checked_sensi_ids":checked_sensi_att_ids,
                   "passed_checked_sensi_atts": checked_sensi_atts, "passed_zip_cate_atts":zip_cate_atts,
                   "passed_binary_names": binary_atts, "passed_binary_ids": binary_att_ids,
                   "passed_cate_ids": cat_atts_ids, "passed_checked_cate_atts":checked_cate_atts,
                   "passed_checked_cate_atts_ids": checked_cate_att_ids, "num_binary_att":num_binary_att,
                   "num_cat_att":num_cat_att, "num_all_att": num_all_att
                   }
        request.session['running_data'] = "unprocessed"

        return render(request, 'rankingfacts/parameters_dash_boot.html', context)
    else: # first access this link without post parameters
        res_ranked_cols = "false"
        context = {'passed_data_name': passed_data_name, "passed_json_columns": json_data_table,
                   "passed_column_name": att_list, "drawable_atts": atts_info["drawable_atts"],
                   "passed_cate_atts": cat_atts,"passed_res_atts": res_ranked_cols,
                   "passed_json_columns_header": json_header_table, "passed_cols_ids": att_ids_list,
                   "passed_ranked_atts": res_ranked_cols,"passed_ranked_atts_ids": res_ranked_cols,
                   "passed_zip_numeric":zip_num_atts,"passed_atts_json": res_ranked_cols,
                   "passed_num_atts_ids": num_atts_ids,"passed_num_atts_names":num_atts_names,
                   "passed_binary_zips": zip_binarys,"passed_checked_sensi_ids":res_ranked_cols,
                   "passed_checked_sensi_atts": res_ranked_cols,"passed_zip_cate_atts":zip_cate_atts,
                   "passed_binary_names": binary_atts, "passed_binary_ids": binary_att_ids,
                   "passed_cate_ids":cat_atts_ids,"passed_checked_cate_atts":res_ranked_cols,
                   "passed_checked_cate_atts_ids": res_ranked_cols, "num_binary_att":num_binary_att,
                   "num_cat_att":num_cat_att, "num_all_att": num_all_att
                   }

        request.session['running_data'] = "unprocessed"
        return render(request, 'rankingfacts/parameters_dash_boot.html', context)

def json_processing_data(request):
    passed_data_name = request.session.get('passed_data_name')

    up_data = DataDescriberUI()
    up_data.read_dataset_from_csv(passed_data_name+".csv")
    up_data.get_json_data()

    total_json = up_data.json_data

    return HttpResponse(total_json, content_type='application/json')

def json_processing_hist(request):
    passed_data_name = request.session.get('passed_data_name')
    description_file = passed_data_name + "_plot.json"
    plot_json = read_json_file(description_file)
    return HttpResponse(json.dumps(plot_json), content_type='application/json')

def json_generate_ranking(request):
    passed_data_name = request.session.get('passed_data_name')

    res_ranking = generateRanking(passed_data_name)

    return HttpResponse(res_ranking, content_type='application/json')


# view function to handle processing parameter settings
def norm_process(request):
    # NOTICE: no .csv suffix in current passed file name
    passed_data_name = request.session.get('passed_data_name')
    # get the string attribute first
    cat_atts = get_categorical_attributes_csv(passed_data_name + ".csv")
    # cleanse the data first, creating a csv named "_norm.csv"
    norm_data = cleanseData(passed_data_name + ".csv", columns_to_exclude=cat_atts)
    norm_data_name = passed_data_name + "_norm"
    norm_data.to_csv(norm_data_name+".csv", index=False)

    atts_info = get_histograms_of(norm_data_name + ".csv")

    # get the categorical attribuet from original uploaded dataset
    cat_atts = atts_info["cate_atts"]
    cat_atts_ids = ["att" + str(i) for i in range(len(cat_atts))]
    zip_cate_atts = zip(cat_atts, cat_atts_ids)

    att_list = atts_info["all_atts"]
    att_ids_list = ["att" + str(i) for i in range(len(att_list))]
    num_atts_names = atts_info["numeric_atts"]
    num_atts_ids = ["att" + str(i) for i in range(len(num_atts_names))]
    zip_num_atts = zip(num_atts_names, num_atts_ids)

    # get the binary attribute from original uploaded dataset
    binary_atts = get_binary_attributes_csv(passed_data_name + ".csv")
    binary_att_ids = ["binary_att" + str(i) for i in range(len(binary_atts))]
    zip_binarys = zip(binary_atts, binary_att_ids)

    row_binary_atts = []
    for bai in range(len(binary_att_ids)):
        if bai % 3 == 0:
            row_binary_atts.append("endrow")
        else:
            row_binary_atts.append("row")
    row_cat_atts = []
    for cai in range(len(cat_atts_ids)):
        if cai % 3 == 0:
            row_cat_atts.append("endrow")
        else:
            row_cat_atts.append("row")
    row_num_atts = []
    for ai in range(len(num_atts_ids)):
        if ai % 3 == 0:
            row_num_atts.append("endrow")
        else:
            row_num_atts.append("row")
    num_binary_att = zip(binary_atts, binary_att_ids, row_binary_atts)
    num_cat_att = zip(cat_atts, cat_atts_ids, row_cat_atts)
    num_all_att = zip(num_atts_names, num_atts_ids, row_num_atts)

    # initialize json data and header for dataTables
    json_data_table = []
    json_header_table = []
    for i in range(len(att_list)):
        json_data_table.append({"data": str(att_list[i])})
        json_header_table.append({"title": str(att_list[i]), "targets": i})

    if request.method == 'POST':
        # if user submit, save user input to server as parameter file
        # get user input
        json_parameter = {}
        included_atts = []
        included_atts_ids = []
        atts_weights = []

        for i in range(len(num_atts_names)):
            atti_id = num_atts_ids[i]
            atti_name = num_atts_names[i]

            atti_json = {}
            att_weight = request.POST[atti_id + "_weight"]
            att_rank = request.POST[atti_id + "_checks"]
            att_order = request.POST[atti_id + "_order"]
            if att_weight == "":  # if empty then use default value 1.0
                att_weight = 1.0
            atti_json["weight"] = att_weight
            atti_json["rank"] = att_rank
            atti_json["order"] = att_order

            if att_rank == atti_name:
                ranked_weight = float(att_weight)
                included_atts.append(atti_name)
                included_atts_ids.append(atti_id)
                if att_order == "lower":
                    ranked_weight = - ranked_weight
                atts_weights.append(ranked_weight)

            json_parameter[atti_id] = atti_json
        json_parameter["ranked_atts"] = included_atts
        json_parameter["ranked_atts_weight"] = atts_weights

        # get checked binary sensitive attribute
        checked_sensi_atts = []
        checked_sensi_att_ids = []
        for vi in range(len(binary_atts)):
            vi_name = binary_atts[vi]
            vi_id = binary_att_ids[vi]
            att_sensi = request.POST[vi_id + "_sensi_checks"]
            if att_sensi == vi_name:
                checked_sensi_atts.append(vi_name)
                checked_sensi_att_ids.append(vi_id)
        json_parameter["checked_sensi_atts"] = checked_sensi_atts

        # get checked categorical sensitive attribute
        checked_cate_atts = []
        checked_cate_att_ids = []
        for vi in range(len(cat_atts)):
            vi_name = cat_atts[vi]
            vi_id = cat_atts_ids[vi]
            att_cate = request.POST[vi_id + "_cate_checks"]
            if att_cate == vi_name:
                checked_cate_atts.append(vi_name)
                checked_cate_att_ids.append(vi_id)
        json_parameter["checked_cate_atts"] = checked_cate_atts

        # save user input parameters to server
        json_parameter_outputfn = norm_data_name + "_rankings.json"
        with open(json_parameter_outputfn, 'w') as outfile:
            json.dump(json_parameter, outfile, indent=4)

        res_ranked_cols = ["GeneratedScore"]
        res_ranked_cols = res_ranked_cols + included_atts
        json_rank_table = []
        json_rank_header_table = []
        for i in range(len(res_ranked_cols)):
            json_rank_table.append({"data": str(res_ranked_cols[i])})
            json_rank_header_table.append({"title": str(res_ranked_cols[i]),"targets": i})

        context = {'passed_data_name': passed_data_name, "passed_json_columns": json_data_table,
                   "passed_column_name": att_list, "drawable_atts": atts_info["drawable_atts"],
                   "passed_cate_atts": cat_atts, "passed_atts_json": json_parameter,
                   "passed_ranked_atts": included_atts, "passed_res_atts": res_ranked_cols,
                   "passed_json_ranks": json_rank_table, "passed_json_columns_header": json_header_table,
                   "passed_cols_ids": att_ids_list, "passed_json_ranks_header": json_rank_header_table,
                   "passed_ranked_atts_ids": included_atts_ids, "passed_zip_numeric": zip_num_atts,
                   "passed_num_atts_ids": num_atts_ids, "passed_num_atts_names": num_atts_names,
                   "passed_binary_zips": zip_binarys,"passed_checked_sensi_ids":checked_sensi_att_ids,
                   "passed_checked_sensi_atts": checked_sensi_atts,"passed_zip_cate_atts":zip_cate_atts,
                   "passed_binary_names": binary_atts, "passed_binary_ids": binary_att_ids,
                   "passed_cate_ids": cat_atts_ids, "passed_checked_cate_atts":checked_cate_atts,
                   "passed_checked_cate_atts_ids": checked_cate_att_ids,"num_binary_att":num_binary_att,
                   "num_cat_att":num_cat_att, "num_all_att": num_all_att
                   }
        request.session['running_data'] = "processed"
        return render(request, 'rankingfacts/parameters_norm_dash_boot.html', context)
    else:
        res_ranked_cols = "false"
        context = {'passed_data_name': passed_data_name, "passed_json_columns": json_data_table,
                   "passed_column_name": att_list, "drawable_atts": atts_info["drawable_atts"],
                   "passed_cate_atts": cat_atts, "passed_res_atts": res_ranked_cols,
                   "passed_json_columns_header": json_header_table, "passed_cols_ids": att_ids_list,
                   "passed_ranked_atts": res_ranked_cols, "passed_ranked_atts_ids": res_ranked_cols,
                   "passed_zip_numeric": zip_num_atts, "passed_atts_json": res_ranked_cols,
                   "passed_num_atts_ids": num_atts_ids, "passed_num_atts_names": num_atts_names,
                   "passed_binary_zips": zip_binarys,"passed_checked_sensi_ids":res_ranked_cols,
                   "passed_checked_sensi_atts": res_ranked_cols,"passed_zip_cate_atts":zip_cate_atts,
                   "passed_binary_names": binary_atts, "passed_binary_ids": binary_att_ids,
                   "passed_cate_ids":cat_atts_ids,"passed_checked_cate_atts":res_ranked_cols,
                   "passed_checked_cate_atts_ids": res_ranked_cols,"num_binary_att":num_binary_att,
                   "num_cat_att":num_cat_att, "num_all_att": num_all_att
                   }

        request.session['running_data'] = "processed"
    return render(request, 'rankingfacts/parameters_norm_dash_boot.html', context)

def norm_json_processing_data(request):
    # NOTICE: input name need to update to _norm version for processing data
    passed_data_name = request.session.get('passed_data_name')

    up_data = DataDescriberUI()
    up_data.read_dataset_from_csv(passed_data_name + "_norm.csv")
    up_data.get_json_data()

    total_json = up_data.json_data

    return HttpResponse(total_json, content_type='application/json')

def norm_json_processing_hist(request):
    # NOTICE: input name need to update to _norm version for processing data
    passed_data_name = request.session.get('passed_data_name')
    description_file = passed_data_name + "_norm_plot.json"
    plot_json = read_json_file(description_file)
    return HttpResponse(json.dumps(plot_json), content_type='application/json')

def norm_json_generate_ranking(request):
    # NOTICE: input name need to update to _norm version for processing data
    passed_data_name = request.session.get('passed_data_name')
    res_ranking = generateRanking(passed_data_name+"_norm")
    return HttpResponse(res_ranking, content_type='application/json')


# view functions to handle results page
def nutrition_facts(request):
    passed_data_name = request.session.get('passed_data_name')
    passed_running_data_flag = request.session.get("running_data")
    # for previous step, return to corresponding parameter setting page
    unprocessed_flag = True

    if passed_running_data_flag == "processed":
        ranks_file = passed_data_name + "_norm_rankings.json"
        cur_data_name = passed_data_name + "_norm"
        unprocessed_flag = False
    else:
        ranks_file = passed_data_name + "_rankings.json"
        cur_data_name = passed_data_name
    # read the parameter file in server that stores all the parameter inputs from user
    rankings_paras = read_json_file(ranks_file)
    chosed_atts = rankings_paras["ranked_atts"]
    checked_sensi_atts = rankings_paras["checked_sensi_atts"]
    checked_cate_atts = rankings_paras["checked_cate_atts"]

    # get the choosed atts and its weight in the parameter file
    att_weights = {}
    for i in range(len(chosed_atts)):
        att_weights[chosed_atts[i]] = rankings_paras["ranked_atts_weight"][i]


    # get size of upload data
    total_n = getSizeOfRanking(cur_data_name)

    # compute statistics of top 10 and overall in generated ranking
    att_stats_topTen = compute_statistic_topN(chosed_atts,cur_data_name,10)
    att_stats_all = compute_statistic_topN(chosed_atts,cur_data_name,total_n)
    # compute top 3 correlated attributes
    att_correlated = compute_correlation(cur_data_name)
    # set the correlation threshold
    low_coef_threshold = 0.25
    high_coef_threshold = 0.75
    # generate the json data for correlation table
    top3_correlated_attts = {}
    for ai in att_correlated:
        ai_coef = abs(ai[0])
        ai_name = ai[1]
        if ai_coef >= high_coef_threshold:
            top3_correlated_attts[ai_name] = [ai_coef,"high"]
        else:
            if ai_coef <= low_coef_threshold:
                top3_correlated_attts[ai_name] = [ai_coef, "low"]
            else:
                top3_correlated_attts[ai_name] = [ai_coef, "median"]

    # compute statistics of top 3 correlated attributes for detailed ingredients widget
    top_corre_atts = [att_correlated[i][1] for i in range(len(att_correlated))]
    corre_att_stats_topTen = compute_statistic_topN(top_corre_atts, cur_data_name, 10)
    corre_att_stats_all = compute_statistic_topN(top_corre_atts, cur_data_name, total_n)


    # compute the slope of generated scores at a specified top-k
    # set the slope threshold for stability
    slope_threshold = 0.25
    if total_n >= 100:
        slope_top_ten = computeSlopeOfScores(cur_data_name,10)
        slope_top_hundred = computeSlopeOfScores(cur_data_name, 100)
        stable_ten = abs(slope_top_ten) <= slope_threshold
        stable_hundred = abs(slope_top_hundred) <= slope_threshold
        stable_res = {"Top-10": stable_ten, "Top-100": stable_hundred}
        slope_overall = "false"
    else:
        if total_n >=10:
            slope_top_ten = computeSlopeOfScores(cur_data_name, 10)
            slope_overall = computeSlopeOfScores(cur_data_name, total_n)
            stable_ten = abs(slope_top_ten) <= slope_threshold
            stable_overall = abs(slope_overall) <= slope_threshold
            slope_top_hundred = "NA"
            stable_res = {"Top-10": stable_ten,"Overall": stable_overall}
        else:
            slope_top_ten = "NA"
            slope_top_hundred = "NA"
            slope_overall = "false"
            stable_res = {}

    # run the fairness validation for three oracles
    fair_all_oracles, fair_res_oracles, alpha_default, top_K = runFairOracles(checked_sensi_atts,cur_data_name)

    checked_cate_att_ids = ["att"+str(i) for i in range(len(checked_cate_atts))]
    # compute the number of pir charts
    pie_n = len(checked_cate_att_ids)
    row_n = int(np.ceil(pie_n/2))
    place_n = int(5 + (row_n-1)*2)
    split_n = int(row_n * 2 +1)

    context = {'passed_data_name': passed_data_name, "passed_att_weights": att_weights,
               "passed_att_stats_topTen": att_stats_topTen, "passed_att_stats_all": att_stats_all,
               "passed_att_correlated": top3_correlated_attts, "passed_unprocessing_flag": unprocessed_flag,
               "corre_att_stats_topTen": corre_att_stats_topTen, "corre_att_stats_all": corre_att_stats_all,
               "passed_fair_all_oracles": fair_all_oracles,"passed_fair_res_oracles":fair_res_oracles,
               "passed_slope_ten":slope_top_ten, "passed_slope_hundred":slope_top_hundred,
               "passed_stable_res":stable_res, "passed_slope_threshold":slope_threshold,
               "passed_alpha_default":alpha_default, "passed_coef_high":high_coef_threshold,
               "passed_top_k":top_K, "passed_coef_low": low_coef_threshold,
               "passed_slope_overall": slope_overall, "passed_pie_att_ids": checked_cate_att_ids,
               "passed_pie_atts": checked_cate_atts, "passed_range_place": range(place_n),
               "passed_range_split": range(split_n),
               }
    return render(request, 'rankingfacts/ranking_facts_widget_boot.html', context)

def json_scatter_score(request):
    passed_data_name = request.session.get('passed_data_name')
    passed_running_data_flag = request.session.get("running_data")
    if passed_running_data_flag == "processed":
        cur_data_name = passed_data_name + "_norm"
    else:
        cur_data_name = passed_data_name

    scatter_data = get_score_scatter(cur_data_name)
    return HttpResponse(json.dumps(scatter_data), content_type='application/json')

def json_piechart_data(request):
    passed_data_name = request.session.get('passed_data_name')
    passed_running_data_flag = request.session.get("running_data")

    if passed_running_data_flag == "processed":
        cur_data_name = passed_data_name + "_norm"
        ranks_file = passed_data_name + "_norm_rankings.json"
    else:
        cur_data_name = passed_data_name
        ranks_file = passed_data_name + "_rankings.json"

    # read the parameter file in server that stores all the parameter inputs from user
    rankings_paras = read_json_file(ranks_file)
    checked_cate_atts = rankings_paras["checked_cate_atts"]
    piechart_json_data = get_chart_data(cur_data_name,checked_cate_atts)

    return HttpResponse(json.dumps(piechart_json_data), content_type='application/json')