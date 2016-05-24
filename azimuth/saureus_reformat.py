import csv
import numpy as np
import pandas as pd
from model_comparison import predict
import scipy.stats

def ranks(scores):
    ranklist = np.arange(len(scores))
    return zip(*sorted(zip(scores, ranklist)))[1]
    
def spearman(scores0, scores1):
    n = len(scores0)
    running_sum = sum([(scores0[i] - scores1[i]) ** 2 for i in xrange(n)])
    return 1.0 - ((6.0 * running_sum) / (1.0 * n * (n**2 - 1)))

with open("2016-03-10 Saueus OnTarget to Microsoft.csv", 'r') as saureus:
    saureus_iter = csv.reader(saureus)
    saureus_data = [data for data in saureus_iter]
data_array = np.asarray(saureus_data)
saureus.close()
(n, k) = data_array.shape
processed_array = [["Construct Barcode", "Construct IDs", "Gene", "Amino Acid", "Pct Pep", "Assay", "Cell Type", "LFC", "Zscore", "NormZscore", "PctRank", "Rev"]]
for i in xrange(3, n):
    for j in xrange(6):
        if data_array[i][k - 6 + j] == "yes_rev":
            processed_row = []
            processed_row.extend(data_array[i][0:5])
            if j == 0:
                processed_row.extend(["6TG", "293T"])
            elif j == 1:
                processed_row.extend(["viability", "293T"])
            elif j == 2:
                processed_row.extend(["6TG", "A375"])
            elif j == 3:
                processed_row.extend(["PLX", "A375"])
            elif j == 4:
                processed_row.extend(["viability", "A375"])
            elif j == 5: 
                processed_row.extend(["viability", "MOLM13"])
            processed_row.append(data_array[i][5 + j])
            processed_row.append(data_array[i][12 + j])
            processed_row.append(data_array[i][19 + j])
            processed_row.append(data_array[i][26 + j])
            processed_row.append("TRUE")
            processed_array.append(processed_row)
        elif data_array[i][k - 6 + j] == "yes_asis":
            processed_row = []
            processed_row.extend(data_array[i][0:5])
            if j == 0:
                processed_row.extend(["6TG", "293T"])
            elif j == 1:
                processed_row.extend(["viability", "293T"])
            elif j == 2:
                processed_row.extend(["6TG", "A375"])
            elif j == 3:
                processed_row.extend(["PLX", "A375"])
            elif j == 4:
                processed_row.extend(["viability", "A375"])
            elif j == 5:
                processed_row.extend(["viability", "MOLM13"])
            processed_row.append(data_array[i][5 + j])
            processed_row.append(data_array[i][12 + j])
            processed_row.append(data_array[i][19 + j])
            processed_row.append(data_array[i][26 + j])
            processed_row.append("FALSE")
            processed_array.append(processed_row)
outfile = np.asarray(processed_array)[1:]
labels = np.asarray(processed_array)[0]

table = pd.DataFrame(data = outfile, columns = labels)
table.to_pickle("processed_saureus.pickle")
table.to_csv("processed_saureus.csv")

"""
test_table = pd.DataFrame.from_csv("FC_plus_RES_withPredictions.csv")
TEST_PREDICT = predict(np.asarray(test_table['30mer'].values, dtype=str), np.asarray(test_table['Amino Acid Cut position'].values, dtype=float), np.asarray(test_table['Percent Peptide'].values, dtype=float))
TEST_SCORE = np.asarray(test_table['score_drug_gene_rank'])
TEST_PREDICTIONS = np.asarray(test_table['predictions'])
print "TEST"
print scipy.stats.spearmanr(TEST_PREDICT, TEST_SCORE)
print scipy.stats.spearmanr(TEST_PREDICT, TEST_PREDICTIONS)
"""

# EEF2 from 0 to 1202
# HNRNPU from 1203 to 2285
# HPRT1 from 2286 to 2431
# MED12 from 2432 to 3365
# NF1 from 3366 to 4211
# NF2 from 4212 to 4514
# NUDT5 from 4515 to 4605
# PELP1 from 4606 to 6924
# TFRC from 6925 to 7464
"""
EEF2_PREDICT = predict(np.asarray(table['Construct IDs'][0:1203].apply(lambda x: x[1:26] + 'G' + x[27:31]).values, dtype=str),
                                                       np.asarray(table["Amino Acid"][0:1203].values, dtype=float),
                                                       np.asarray(table["Pct Pep"][0:1203].values, dtype=float))
EEF2_SCORE = np.asarray(table["PctRank"][0:1203])
print "EEF2"
print scipy.stats.spearmanr(EEF2_PREDICT, EEF2_SCORE)
HNRNPU_PREDICT = predict(np.asarray(table['Construct IDs'][1203:2286].apply(lambda x: x[1:26] + 'G' + x[27:31]).values, dtype=str),
                                                       np.asarray(table["Amino Acid"][1203:2286].values, dtype=float),
                                                       np.asarray(table["Pct Pep"][1203:2286].values, dtype=float))
HNRNPU_SCORE = np.asarray(table["PctRank"][1203:2286])
print "HNRNPU"
print scipy.stats.spearmanr(HNRNPU_PREDICT, HNRNPU_SCORE)
HPRT1_PREDICT = predict(np.asarray(table['Construct IDs'][2286:2432].apply(lambda x: x[1:26] + 'G' + x[27:31]).values, dtype=str),
                                                       np.asarray(table["Amino Acid"][2286:2432].values, dtype=float),
                                                       np.asarray(table["Pct Pep"][2286:2432].values, dtype=float))
HPRT1_SCORE = np.asarray(table["PctRank"][2286:2432])
print "HPRT1"
print scipy.stats.spearmanr(HPRT1_PREDICT, HPRT1_SCORE)
MED12_PREDICT = predict(np.asarray(table['Construct IDs'][2432:3366].apply(lambda x: x[1:26] + 'G' + x[27:31]).values, dtype=str),
                                                       np.asarray(table["Amino Acid"][2432:3366].values, dtype=float),
                                                       np.asarray(table["Pct Pep"][2432:3366].values, dtype=float))
MED12_SCORE = np.asarray(table["PctRank"][2432:3366])
print "MED12"
print scipy.stats.spearmanr(MED12_PREDICT, MED12_SCORE)
NF1_PREDICT = predict(np.asarray(table['Construct IDs'][3366:4212].apply(lambda x: x[1:26] + 'G' + x[27:31]).values, dtype=str),
                                                       np.asarray(table["Amino Acid"][3366:4212].values, dtype=float),
                                                       np.asarray(table["Pct Pep"][3366:4212].values, dtype=float))
NF1_SCORE = np.asarray(table["PctRank"][3366:4212])
print "NF1"
print scipy.stats.spearmanr(NF1_PREDICT, NF1_SCORE)
NF2_PREDICT = predict(np.asarray(table['Construct IDs'][4212:4515].apply(lambda x: x[1:26] + 'G' + x[27:31]).values, dtype=str),
                                                       np.asarray(table["Amino Acid"][4212:4515].values, dtype=float),
                                                       np.asarray(table["Pct Pep"][4212:4515].values, dtype=float))
NF2_SCORE = np.asarray(table["PctRank"][4212:4515])
print "NF2"
print scipy.stats.spearmanr(NF2_PREDICT, NF2_SCORE)
"""
for i in xrange(7):
    NUDT5_PREDICT = predict(np.asarray(table['Construct IDs'].apply(lambda x: x[i:25+i] + 'GG' + x[27+i:30+i]).values, dtype=str),
                                                           np.asarray(table["Amino Acid"].values, dtype=float),
                                                           np.asarray(table["Pct Pep"].values, dtype=float))
    NUDT5_SCORE = np.asarray(table["LFC"].values, dtype=float)
    # print "NUDT5"
    print i
    print scipy.stats.spearmanr(NUDT5_PREDICT, NUDT5_SCORE)
"""
PELP1_PREDICT = predict(np.asarray(table['Construct IDs'][4606:6925].apply(lambda x: x[1:26] + 'G' + x[27:31]).values, dtype=str),
                                                       np.asarray(table["Amino Acid"][4606:6925].values, dtype=float),
                                                       np.asarray(table["Pct Pep"][4606:6925].values, dtype=float))
PELP1_SCORE = np.asarray(table["PctRank"][4606:6925])
print "PELP1"
print scipy.stats.spearmanr(PELP1_PREDICT, PELP1_SCORE)
TFRC_PREDICT = predict(np.asarray(table['Construct IDs'][6925:7465].apply(lambda x: x[1:26] + 'G' + x[27:31]).values, dtype=str),
                                                       np.asarray(table["Amino Acid"][6925:7465].values, dtype=float),
                                                       np.asarray(table["Pct Pep"][6925:7465].values, dtype=float))
TFRC_SCORE = np.asarray(table["PctRank"][6925:7465])
print "TFRC"
print scipy.stats.spearmanr(TFRC_PREDICT, TFRC_SCORE)
TOTAL_PREDICT = predict(np.asarray(table['Construct IDs'].apply(lambda x: x[1:26] + 'G' + x[27:31]).values, dtype=str),
                                                       np.asarray(table["Amino Acid"].values, dtype=float),
                                                       np.asarray(table["Pct Pep"].values, dtype=float))
TOTAL_SCORE = np.asarray(table["LFC"].values, dtype=float)
print "TOTAL"
print scipy.stats.spearmanr(TOTAL_PREDICT, TOTAL_SCORE)
"""
