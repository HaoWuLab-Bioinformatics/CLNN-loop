import numpy as np
import math
import heapq
from itertools import combinations, combinations_with_replacement, permutations
import copy
from itertools import product
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import argparse
import re, sys, os, platform
from collections import Counter
import itertools

def readwrite(inf, outf):
    f = open(inf, 'r')
    out = open(outf, 'w')

    for line in f.readlines():
        list_line = list(line)
        if list_line[0] == '>':
            continue

        length = len(list_line)
        for i in range(length):
            letter = list_line[i]
            if letter == 'A' or letter == 'a':
                out.writelines('0')
            elif letter == 'T' or letter == 't':
                out.writelines('1')
            elif letter == 'G' or letter == 'g':
                out.writelines('2')
            elif letter == 'C' or letter == 'c':
                out.writelines('3')
            out.writelines(' ')

        out.writelines('\n')

    f.close()
    out.close()


def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData


###########################feature NPSE code##############################
def NPSE(k):
    number = len(data)
    length = len(data[0])
    feature_NPSE = np.array([[0.0] * (4 * 4 * (k + 1))] * number)
    for i in range(number):
        for s in range(k + 1):
            for j in range(length - (s + 1)):
                pos = int(data[i][j] * 4 + data[i][j + (s + 1)])
                feature_NPSE[i][pos + (4 * 4 * s)] = feature_NPSE[i][pos + (4 * 4 * s)] + 1 / (length - (s + 1))
    return feature_NPSE


###########################feature PCSF code##############################
def PWM(x, k):
    b = 1 / (4 ** k)
    pwm = []
    length = len(x[0])
    realcounts = np.zeros((length - k + 1, 4 ** k))
    # for i in range(length-k+1):
    # name
    pwm = copy.deepcopy(realcounts)
    totalcounts = []
    psetotalcounts = []
    for s in range(len(x)):
        for i in range(length - k + 1):
            kmer = x[s][i:i + k]
            pos = 0
            for m in range(k):
                pos = int(kmer[m] * (4 ** (k - m - 1))) + pos
            realcounts[i][pos] += 1
    for i in range(length - k + 1):
        totalcounts.append(sum(realcounts[i]))
        psetotalcounts.append(math.sqrt(totalcounts[i]))

    for i in range(length - k + 1):
        for j in range(4 ** k):
            pwm[i][j] = (realcounts[i][j] + b * psetotalcounts[i]) / (totalcounts[i] + psetotalcounts[i])
    return pwm


def PCSF(x, k, sites, pwm):
    b = 1 / (4 ** k)
    pcsf = []
    for s in range(len(x)):
        #length = len(x[s])
        length = 42
        pcsf1 = []
        for i in range(length + 1 - k):
            if i in sites:
                kmer = x[s][i:i + k]
                pos = 0
                for m in range(k):
                    pos = int(kmer[m] * (4 ** (k - m - 1))) + pos
                pcsf1.append(math.log(pwm[i][pos] / b, math.e))
            else:
                pass
        pcsf.append(pcsf1)
    return pcsf


###########################feature RCKmer code##############################

def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer


def RC(kmer):
    myDict = {
        'A': 'T',
        'C': 'G',
        'G': 'C',
        'T': 'A'
    }
    return ''.join([myDict[nc] for nc in kmer[::-1]])


def generateRCKmer(kmerList):
    rckmerList = set()
    myDict = {
        'A': 'T',
        'C': 'G',
        'G': 'C',
        'T': 'A'
    }
    for kmer in kmerList:
        rckmerList.add(sorted([kmer, ''.join([myDict[nc] for nc in kmer[::-1]])])[0])
    return sorted(rckmerList)


def RCKmer(fastas, normalize=True):
    k = 5
    encoding = []
    header = ['#', 'label']
    NA = 'ACGT'

    tmpHeader = []  # all kmer
    for kmer in itertools.product(NA, repeat=k):
        tmpHeader.append(''.join(kmer))
    header = header + generateRCKmer(tmpHeader)
    myDict = {}
    for kmer in header[2:]:
        rckmer = RC(kmer)
        if kmer != rckmer:
            myDict[rckmer] = kmer

    encoding.append(header)
    f = open(fastas, 'r')
    for i in f.readlines():
        list_line = list(i)
        if list_line[0] == '>':
            continue
        else:
            if i[-1] == '\n':
                sequence = i[0:-1]
            else:
                sequence = i
            kmers = kmerArray(sequence, k)  # len=38
            for j in range(len(kmers)):
                if kmers[j] in myDict:
                    kmers[j] = myDict[kmers[j]]
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            # code = [name, label]
            code = []
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            code = np.array(code)
            encoding.append(code)
    return encoding


###########################feature PSTNPss code##############################

def CalculateMatrix_PSTNPss(data, order):
    matrix = np.zeros((len(data[0]) - 2, 64))
    for i in range(len(data[0]) - 2):  # position
        for j in range(len(data)):
            if re.search('-', data[j][i:i + 3]):
                pass
            else:
                matrix[i][order[data[j][i:i + 3]]] += 1
    return matrix


def PSTNPss(inf, sequence_all, cell_line):
    positive = []  # load
    negative = []  # load
    for i in range(len(sequence_all)):
        if cell_line == 'IMR90/':
            positive.append(sequence_all[i])
        else:
            if i < len(sequence_all) / 2:
                positive.append(sequence_all[i])
            else:
                negative.append(sequence_all[i])

    encodings = []
    header = ['#', 'label']
    fastas = []
    f = open(inf, 'r')
    for i in f.readlines():
        list_line = list(i)
        if list_line[0] == '>':
            continue
        else:
            if i[-1] == '\n':
                sequence = i[0:-1]
            else:
                sequence = i
            fastas.append(sequence)
    for pos in range(len(fastas[0]) - 2):
        header.append('Pos.%d' % (pos + 1))
    encodings.append(header)

    '''for i in range(int(len(fastas)/2)):
        positive.append(fastas[i])
        negative.append(fastas[i+97])
    print(len(positive))'''

    nucleotides = ['A', 'C', 'G', 'T']
    trinucleotides = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
    order = {}
    for i in range(len(trinucleotides)):
        order[trinucleotides[i]] = i
    matrix_po = CalculateMatrix_PSTNPss(positive, order)
    if cell_line != 'IMR90/':
        matrix_ne = CalculateMatrix_PSTNPss(negative, order)
    # matrix_po=load
    # matrix_ne=load

    positive_number = len(positive)
    if cell_line != 'IMR90/':
        negative_number = len(negative)
    count = 0
    for i in fastas:
        count += 1
        sequence = i
        code = []
        if cell_line == 'IMR90/':
            for j in range(len(sequence) - 2):
                if re.search('-', sequence[j: j + 3]):
                    code.append(0)
                else:
                    p_num = positive_number
                    po_number = matrix_po[j][order[sequence[j: j + 3]]]
                    if count < int(len(fastas)) and po_number > 0:
                        po_number -= 1
                        p_num -= 1
                    code.append(po_number/p_num)
                    # print(sequence[j: j+3], order[sequence[j: j+3]], po_number, p_num, ne_number, n_num)
            encodings.append(code)
        else:
            for j in range(len(sequence) - 2):
                if re.search('-', sequence[j: j + 3]):
                    code.append(0)
                else:
                    p_num, n_num = positive_number, negative_number
                    po_number = matrix_po[j][order[sequence[j: j + 3]]]
                    ne_number = matrix_ne[j][order[sequence[j: j + 3]]]
                    if count < int(len(fastas) / 2) and po_number > 0:
                        po_number -= 1
                        p_num -= 1
                    if count >= int(len(fastas) / 2) and ne_number > 0:
                        ne_number -= 1
                        n_num -= 1
                    code.append(po_number / p_num - ne_number / n_num)
                    # print(sequence[j: j+3], order[sequence[j: j+3]], po_number, p_num, ne_number, n_num)
            encodings.append(code)
    return encodings


###########################feature PSTNPds code##############################
def CalculateMatrix_PSTNPds(data, order):
    matrix = np.zeros((len(data[0]) - 2, 8))
    for i in range(len(data[0]) - 2):  # position
        for j in range(len(data)):
            if re.search('-', data[j][i:i + 3]):
                pass
            else:
                matrix[i][order[data[j][i:i + 3]]] += 1
    return matrix


def PSTNPds(inf, sequence_all, cell_line):
    sequence_positive = []  # load
    sequence_negative = []  # load
    for i in range(len(sequence_all)):
        if cell_line == 'IMR90/':
            sequence_positive.append(sequence_all[i])
        else:
            if i < len(sequence_all) / 2:
                sequence_positive.append(sequence_all[i])
            else:
                sequence_negative.append(sequence_all[i])

    fastas = []
    f = open(inf, 'r')
    for i in f.readlines():
        list_line = list(i)
        if list_line[0] == '>':
            continue
        else:
            if i[-1] == '\n':
                sequence = i[0:-1]
            else:
                sequence = i
            fastas.append(sequence)

    fastas_new = []
    positive = []
    negative = []
    for i in fastas:
        i = re.sub('T', 'A', i)
        i = re.sub('G', 'C', i)
        fastas_new.append(i)
    for i in sequence_positive:
        i = re.sub('T', 'A', i)
        i = re.sub('G', 'C', i)
        positive.append(i)
    for i in sequence_negative:
        i = re.sub('T', 'A', i)
        i = re.sub('G', 'C', i)
        negative.append(i)
    encodings = []
    header = ['#', 'label']
    for pos in range(len(fastas_new[0]) - 2):
        header.append('Pos.%d' % (pos + 1))
    encodings.append(header)

    '''for i in range(int(len(fastas_new) / 2)):
        positive.append(fastas_new[i])
        negative.append(fastas_new[i + 97])'''

    nucleotides = ['A', 'C']
    trinucleotides = [n1 + n2 + n3 for n1 in nucleotides for n2 in nucleotides for n3 in nucleotides]
    order = {}
    for i in range(len(trinucleotides)):
        order[trinucleotides[i]] = i
    matrix_po = CalculateMatrix_PSTNPds(positive, order)
    if cell_line != 'IMR90/':
        matrix_ne = CalculateMatrix_PSTNPds(negative, order)
    # matrix_po=load
    # matrix_ne=load

    positive_number = len(positive)
    if cell_line != 'IMR90/':
        negative_number = len(negative)
    count = 0
    for i in fastas_new:
        count += 1
        sequence = i
        code = []
        if cell_line == 'IMR90/':
            for j in range(len(sequence) - 2):
                if re.search('-', sequence[j: j + 3]):
                    code.append(0)
                else:
                    p_num = positive_number
                    po_number = matrix_po[j][order[sequence[j: j + 3]]]
                    if count < int(len(fastas)) and po_number > 0:
                        po_number -= 1
                        p_num -= 1
                    code.append(po_number/p_num)
                    # print(sequence[j: j+3], order[sequence[j: j+3]], po_number, p_num, ne_number, n_num)
            encodings.append(code)
        else:
            for j in range(len(sequence) - 2):
                if re.search('-', sequence[j: j + 3]):
                    code.append(0)
                else:
                    p_num, n_num = positive_number, negative_number
                    # p_num = positive_number
                    po_number = matrix_po[j][order[sequence[j: j + 3]]]
                    ne_number = matrix_ne[j][order[sequence[j: j + 3]]]
                    if count < int(len(fastas_new) / 2) and po_number > 0:
                        po_number -= 1
                        p_num -= 1
                    if count >= int(len(fastas_new) / 2) and ne_number > 0:
                        ne_number -= 1
                        n_num -= 1
                    code.append(po_number / p_num - ne_number / n_num)
                    # print(sequence[j: j+3], order[sequence[j: j+3]], po_number, p_num, ne_number, n_num)
            encodings.append(code)
    return encodings


########################### extract first sequence feature##############################
filename = 'data/K562/K562_RF_R_train_new'
database_file = 'K562_RF_R'
cell_line = 'K562/'

database = 'data/'+ cell_line + database_file + '_train_new.fasta'
database_out = 'data/'+ cell_line + database_file + '_train_new.txt'
inf = filename + '.fasta'
outf = filename + '.txt'
readwrite(inf, outf)
readwrite(database, database_out)
k = 5

encodings = RCKmer(inf)
encodings = np.array(encodings[1:])
# np.savetxt('fea.txt', encodings)
feature1_1 = encodings

data = np.loadtxt(outf)
if data.ndim == 1:
    data = np.expand_dims(data, 0)
sites = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
base = np.loadtxt(database_out)
pwm1 = PWM(base, k)  # load
#pwm1 = np.loadtxt('K562_RF_R_PWM.txt')
#np.savetxt('K562_RF_R_PWM.txt',pwm1)
pcsf = PCSF(data, k, sites, pwm1)
pcsf = np.array(pcsf)
feature_pcsf = np.zeros((len(data), 1))
for i in range(len(data)):
    feature_pcsf[i] = pcsf[i].sum()
# np.savetxt('PCSF_.txt',feature_pcsf)
feature2_1 = feature_pcsf

#database = 'K562_RF_R_train_new.fasta'
sequence_all = []
f = open(database, 'r')
for i in f.readlines():
    list_line = list(i)
    if list_line[0] == '>':
        continue
    else:
        if i[-1] == '\n':
            sequence = i[0:-1]
        else:
            sequence = i
        sequence_all.append(sequence)
encodings = PSTNPds(inf, sequence_all, cell_line)
encodings = np.array(encodings[1:])
x3_1 = encodings

encodings = PSTNPss(inf, sequence_all, cell_line)
encodings = np.array(encodings[1:])
x4_1 = encodings

data = np.loadtxt(outf)
if data.ndim == 1:
    data = np.expand_dims(data, 0)

feature5_1 = NPSE(k)
# np.savetxt('NPSE_.txt',feature2)

########################### extract second sequence feature##############################
filename = 'data/K562/K562_RF_F_train_new'
database_file = 'K562_RF_F'
cell_line = 'K562/'

database = 'data/'+ cell_line + database_file + '_train_new.fasta'
database_out = 'data/'+ cell_line + database_file + '_train_new.txt'
inf = filename + '.fasta'
outf = filename + '.txt'
readwrite(inf, outf)
readwrite(database, database_out)

encodings = RCKmer(inf)
encodings = np.array(encodings[1:])
# np.savetxt('fea.txt', encodings)
feature1_2 = encodings

data = np.loadtxt(outf)
if data.ndim == 1:
    data = np.expand_dims(data, 0)
sites = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
base = np.loadtxt(database_out)
pwm1 = PWM(base, k)  # load
#pwm1 = np.loadtxt('K562_RF_F_PWM.txt')
#np.savetxt('K562_RF_F_PWM.txt',pwm1)
pcsf = PCSF(data, k, sites, pwm1)
pcsf = np.array(pcsf)
feature_pcsf = np.zeros((len(data), 1))
for i in range(len(data)):
    feature_pcsf[i] = pcsf[i].sum()
# np.savetxt('PCSF_.txt',feature_pcsf)
feature2_2 = feature_pcsf

sequence_all = []
f = open(database, 'r')
for i in f.readlines():
    list_line = list(i)
    if list_line[0] == '>':
        continue
    else:
        if i[-1] == '\n':
            sequence = i[0:-1]
        else:
            sequence = i
        sequence_all.append(sequence)
encodings = PSTNPds(inf, sequence_all, cell_line)
encodings = np.array(encodings[1:])
x3_2 = encodings

encodings = PSTNPss(inf, sequence_all, cell_line)
encodings = np.array(encodings[1:])
x4_2 = encodings

data = np.loadtxt(outf)
if data.ndim == 1:
    data = np.expand_dims(data, 0)
feature5_2 = NPSE(k)


# np.savetxt('NPSE_.txt',feature2)

#####################################  fuse features  #######################
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData


def load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1, feature5_2):
    feature2_1 = noramlization(feature2_1.reshape(-1, 1))
    feature2_2 = noramlization(feature2_2.reshape(-1, 1))
    feature3_1 = np.zeros(len(feature2_1))
    feature3_2 = np.zeros(len(feature2_1))
    feature4_1 = np.zeros(len(feature2_1))
    feature4_2 = np.zeros(len(feature2_1))
    for i in range(len(feature2_1)):
        feature3_1[i] = x3_1[i].sum()
        feature3_2[i] = x3_2[i].sum()
        feature4_1[i] = x4_1[i].sum()
        feature4_2[i] = x4_2[i].sum()
    x = np.concatenate(
        (feature1_1, feature1_2, feature2_1, feature2_2, feature3_1.reshape(-1, 1),
         feature3_2.reshape(-1, 1), feature4_1.reshape(-1, 1), feature4_2.reshape(-1, 1), feature5_1, feature5_2),
        axis=1)
    return x


x = load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1, feature5_2)

x = np.expand_dims(x, 2)
np.save('test.npy',x)