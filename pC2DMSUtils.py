# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 09:59:24 2015

@author: Taran Driver
"""

import numpy as np
import os
import scipy.io


# from random import randint

def adjustParentIon(path, parentMZ, parentCharge):
    """Manually adjust the parent ion details (charge and m/z) of all mgf files
    in the folder 'path', and write to a new mgf file in the same folder.
    path and parentCharge are strings (parentCharge needs value and polarity
    e.g. '3+'), parentMZ is a float."""
    fileNames = [x for x in os.listdir(path) if \
                 (os.path.isfile(os.path.join(path, x)) and x.split('.')[-1] == 'mgf')]

    for fileName in fileNames:
        fileLines = []
        with open(os.path.join(path, fileName)) as f1:
            for line1 in f1:
                fileLines.append(line1)

        f2 = open(os.path.join(path, ''.join(fileName.split('.')[:-1]) + \
                               '_manuallyAdjustedParentMZ.mgf'), 'w')
        for line2 in fileLines:
            first = line2.split('=')[0]
            if first == 'CHARGE':
                continue  # so you don't write a potentially incorrect charge
                # state calculated by Thermo software itself from raw data
            elif first == 'PEPMASS':
                f2.write('PEPMASS=' + str(parentMZ) + '\n')
                f2.write('CHARGE=' + parentCharge + '\n')
            else:
                f2.write(line2)  # new line is in here already
        f2.close()


def bsResamps(numScans, numResamp):
    "Returns numpy array of (resamples, scans in resample)."
    #    return np.array([[randint(0, numScans-1) for x in range(numScans)] for y\
    #     in range(numResamp)]) #this can be vectorised, as it is below
    return np.random.randint(0, numScans, size=(numResamp, numScans))


def circList(r):
    return [[x, y] for x in range(r + 1) for y in range(r + 1) if x ** 2 + y ** 2 <= r ** 2]


def clearCirc(array, r, c, circList, setVal=0):
    for x in circList:
        try:
            array[r + x[0], c + x[1]] = setVal
        except:
            pass
        try:
            array[r + x[0], c - x[1]] = setVal
        except:
            pass
        try:
            array[r - x[0], c + x[1]] = setVal
        except:
            pass
        try:
            array[r - x[0], c - x[1]] = setVal
        except:
            pass


def covXI(scanList, varList):
    'Compute the covariance between a scan list and a single variable'
    assert len(varList) == scanList.shape[0]

    numScans = len(varList)
    numSlices = scanList.shape[1]

    SiSx = (varList.sum(axis=0)) * (scanList.sum(axis=0))  # can be kept in
    # np.array format because this is multiplication of array by scalar

    Six = np.zeros(numSlices)

    for scanIndex in range(numScans):  # scanIndex corresponds to scan
        # number (scanIndex+1)

        Six += varList[scanIndex] * scanList[scanIndex]  # Six for the
        # single scan indexed by 'scanindex'. Can be kept in np.array format
        # because this is multiplication of array by scalar.

    return (Six / (numScans - 1)) - (SiSx / (numScans * (numScans - 1)))


def createArrayParams(scanfolder):
    "created for when pre-existing arrays have been binned/sampled etc.\
    and you want to create new scan folder"
    # The text file that gets read in reports a min mz and max mz and then
    # gives intensity readings at each m/z value. The min m/z value at
    # which is gives an intensity reading for is one sliceSize higher than
    # the reported min m/z, the max m/z value that it gives an intensity
    # reading for is the same as the reported max m/z. This maintains this
    # format
    if not os.path.isfile(scanfolder + '/array.npy'):
        raise ValueError('no array.npy saved in ' + scanfolder)
    array = np.load(scanfolder + '/array.npy', allow_pickle=True)
    slicesize = np.average(np.diff(array[0, :]))
    params = np.zeros(8)

    params[0] = array.shape[1]  # number of m/z bins/slices
    params[1] = array.shape[0] - 1  # number of scans
    params[2] = array.shape[0] - 1  # number of scans, historic duplicate
    params[3] = slicesize
    params[4] = 0.0  # historic (interval for interpolation object, now redundant)
    params[5] = array[0, 0] - slicesize  # min m/z, see above
    params[6] = array[0, -1]  # max m/z
    params[7] = slicesize  # historic duplicate

    np.save(scanfolder + '/array_parameters.npy', params)


def cutAC(arrayToCut, pixAcross=8, threshFrac=1e-6):
    """Cuts along diagonal (x=y) line that is pixAcross pixels wide of a 
    SQUARE array (array is covariance/partial covariance map, in which case 
    the x=y line is the autocorrelation (AC) line). All values above the 
    threshold value (= threshFrac *  max value of array) along this line 
    are set to the threshold value. 
    Runs quicker with these 3 clumsy loops than with 'try' clause.
    Returns the cut array."""

    array = np.copy(arrayToCut)  # copy otherwise it changes the array in place

    thresh = np.nanmax(array) * threshFrac
    arrayWidth = len(array)

    for row in range(pixAcross, (arrayWidth - pixAcross)):
        for pixel in range((row - pixAcross), (row + (pixAcross + 1))):
            if array[row, pixel] > thresh:
                array[row, pixel] = thresh

    for row2 in range((arrayWidth - pixAcross), arrayWidth):
        for col in range((row2 - pixAcross), arrayWidth):
            if array[row2, col] > thresh:
                array[row2, col] = thresh

    for row3 in range(0, pixAcross):
        for col3 in range(0, (row3 + (pixAcross + 1))):
            if array[row3, col3] > thresh:
                array[row3, col3] = thresh

    return array


def sampleSpectra(array_orig, samp_step):
    "take every samp_step m/z bin, starting with the zeroeth"
    return array_orig[:, np.arange(0, array_orig.shape[1], samp_step)]


def binSpectra(array_orig, bin_size):
    "bins, then divides each bin by the number of elements put into it. Each\
    bin is of same size in this implementation"
    quot, rem = divmod(array_orig.shape[1], bin_size)
    array_orig = array_orig[:, :bin_size * quot]  # 'trim'so that reshaping works

    return array_orig.reshape(array_orig.shape[0], quot, bin_size).sum(axis=2) \
        / float(bin_size)


def isInFeatList(mzs, featList, ionThresh=1.5, withNumber=False):
    """Is an m/z pair in a given list of features. mzs is tuple, length 2."""
    if not (isinstance(mzs, tuple) and len(mzs) == 2):
        raise TypeError('mzs must be tuple of length 2')
    mz1, mz2 = sorted(mzs)
    compFeats = [sorted((x[0], x[1])) for x in featList]

    num = -1  # number of feature (in Python indexing, so if it is the first
    # feature then 0 is returned)
    for y in compFeats:
        num += 1
        if abs(mz1 - y[0]) <= ionThresh and abs(mz2 - y[1]) <= ionThresh:
            if withNumber:
                return True, num
            else:
                return True
    # this only executes if a 'return' condition hasn't been met above
    # (i.e. the feature not found)
    if withNumber:
        return False, np.nan  # so that you can generally call
        # "isIn,num=isInFeatList(...,...,withNumber=False)"
    else:
        return False


def maxIndices(array):
    """Returns row and column index of the maximum value in a 2D array"""
    posMax = np.unravel_index(array.argmax(), array.shape)
    return posMax[0], posMax[1]


def minIndices(array):
    """Returns row and column index of the minimum value in a 2D array"""
    posMin = np.unravel_index(array.argmin(), array.shape)
    return posMin[0], posMin[1]


def readmgf(path, maxScans=1):
    scanCountA = 0
    with open(path) as f:
        for i, lineA in enumerate(f):
            if lineA == 'END IONS\n' or lineA == 'END IONS':
                scanCountA += 1
                if scanCountA >= maxScans:
                    break
    f.close()

    array = np.zeros((i, 2)) * np.nan
    scanCount = 0
    count = 0
    inIons = False

    with open(path) as f:
        for line in f:
            if line == 'END IONS\n' or line == 'END IONS':  # if there is only one
                # scan in the mgf file (maybe only if I've manually deleted the
                # other scans), the new line character ('\n') is lost.
                inIons = False
                scanCount += 1
                if scanCount >= maxScans:
                    break
                count += 1

            elif inIons:
                lspl = line.split()
                array[count, 0], array[count, 1] = float(lspl[0]), float(lspl[1])
                count += 1

            elif line != 'BEGIN IONS\n' and '=' not in line:
                inIons = True
                lspl = line.split()
                array[count, 0], array[count, 1] = float(lspl[0]), float(lspl[1])
                count += 1
    f.close()

    return array[:count, :]


def removeACFeats(featList, acThresh=5.5):
    'Removes all features closer than or equal to acThresh Da in m/z. Returns numpy array.'
    """Takes list or numpy array as featList"""
    featList = np.asarray(featList)  # ensures input is numpy array to allow the
    # fancy indexing below
    return featList[np.array([(abs(feat[0] - feat[1]) > acThresh) for feat in \
                              featList])]


def saveMatFile(array, saveTo, fieldName='array'):
    'No need to specify .mat extension'
    scipy.io.savemat(saveTo, {fieldName: array})


def saveSyx(scanFolder, numScansList=['all']):
    """Calculate Syx and save in scan folder, this is a computationally
    heavy operation and it is useful to have the result saved for further 
    analyses."""

    numScansList.sort()

    numScansListStr = [str(x) for x in numScansList]  # make a list of strings
    # of the numbers in numScansList, to enable printing below
    print('Performing saveSyx for ' + scanFolder + ' for ' + \
          ', '.join(numScansListStr) + ' scans')

    params = np.load(scanFolder + '/array_parameters.npy', allow_pickle=True)

    if 'all' in numScansList:
        numScansList.remove('all')
        numScansList.append(int(params[1]))
        print('(all scans = ' + str(int(params[1])) + ')')

    numSlices = int(params[0])  # required to declare Syx
    Syx = np.zeros((numSlices, numSlices))
    array = np.load(scanFolder + '/array.npy', allow_pickle=True)

    for scan in range(1, max(numScansList) + 1):  # row 0 of array holds
        # m/z values so start from from 1
        spectrumX = np.matrix(array[scan, :])  # require it to be matrix for
        # subsequent matrix multiplication. By default here, this is a row
        # vector.
        Syx += np.matrix.transpose(spectrumX) * spectrumX  # column vector * row
        # vector -> outer product

        if scan % 100 == 0:
            print('Syx calculated for first ' + str(scan) + ' scans')

        if scan in numScansList:
            np.save(scanFolder + '/Syx_' + str(scan) + '_scans.npy', Syx)
            print('Syx saved for ' + str(scan) + ' scans')

    print('Completed saveSyx for ' + scanFolder + ' for ' + \
          ', '.join(numScansListStr) + ' scans')
    if 'all' in numScansListStr:
        print('(all scans = ' + str(int(params[1])) + ')')

    return


def saveSyxEinSum(scanFolder, numScans=13):
    """Calculate Syx and save in scan folder, this is a computationally
    heavy operation and it is useful to have the result saved for further 
    analyses."""

    print('Performing saveSyxEinSum for ' + scanFolder + ' for ' + str(numScans) + \
          ' scans')

    if numScans == 'all':
        params = np.load(scanFolder + '/array_parameters.npy', allow_pickle=True)
        numScans = int(params[1])
        print('(all scans = ' + str(numScans) + ')')

    array = np.load(scanFolder + '/array.npy', allow_pickle=True)

    Syx = np.einsum('ij,ik->jk', array[1:numScans + 1], array[1:numScans + 1])
    # row 0 of array holds m/z values so start from from 1
    np.save(scanFolder + '/Syx_' + str(numScans) + '_scans.npy', Syx)

    print('Completed saveSyxEinSum for ' + scanFolder + ' for ' + str(numScans) + \
          ' scans')
    if numScans == 'all':
        print('(all scans = ' + str(int(params[1])) + ')')

    return


def scaleToPower(array, power=0.25):
    """Scales an array to a given power"""

    arrayabs = abs(array)
    arraysign = arrayabs / array
    arraysign[np.isnan(arraysign)] = 1.0
    arrayscaled = arrayabs ** power

    return arrayscaled * arraysign


def sortList(listToSort, sortCol=3):
    'Default sortCol is 3 (significance). Returns numpy array.'
    """Takes list or numpy array as listToSort"""
    scores = [entry[sortCol] for entry in listToSort]
    return np.array([listToSort[index] for index in \
                     reversed(np.argsort(scores))])


def varII(varList):
    'Compute the variance of a single variable, i'
    numScans = len(varList)
    Si2 = (varList ** 2).sum(axis=0)
    S2i = varList.sum(axis=0) ** 2

    return (Si2 - S2i / numScans) / (numScans - 1)


def topNfilt(array, topN, binSize=100):
    'Performs top N filtering on numpy nx2 array (1st col m/z, 2nd col ints)'
    """Doesn't require array to be sorted"""
    minMZ = np.nanmin(array[:, 0])
    maxMZ = np.nanmax(array[:, 0])
    binFloor = minMZ  # this could be changed, but currently defines bin edges
    # according to min m/z in data

    filtList = np.zeros((1, 2))  # initialise array so that it can be appended to
    binFloor = minMZ

    while (binFloor + binSize) <= maxMZ:
        inBin = array[(array[:, 0] >= binFloor) & (array[:, 0] < binFloor + binSize)]
        filtList = np.append(filtList, inBin[inBin[:, 1].argsort()][-topN:, :], \
                             axis=0)
        binFloor += binSize

    finalTopN = int(np.around(topN * ((maxMZ - binFloor) / np.float(binSize))))  # no.
    # of peaks to take from last bin is normalised according to how wide
    # last bin is
    inLastBin = array[(array[:, 0] >= binFloor)]
    filtList = np.append(filtList, \
                         inLastBin[inLastBin[:, 1].argsort()][-finalTopN:, :], axis=0)

    filtList = filtList[1:, :]  # remove the first zeros used to initialise

    return filtList[filtList[:, 0].argsort()]  # sort according to m/z


# %%
"""These functions all work but need some serious attention!"""


def byPosIons(peptide_sequence):
    'Peptide sequence requires PTMs in parentheses after modified residue'
    # Function to calculate the monoisotopic mass of a peptide sequence,
    # its peptide ions, and its sequence b and y ions

    # Dictionary of monoisotopic mass changes due to PTMs, to 4 or 5 sf.
    # Can be added to
    PTM_monoisotopic_mass_changes = {'p': 79.9663, 'Me': 14.0157, 'Ac': 42.0106,
                                     'pm': 238.2297, 's': 79.95682,
                                     'NH2': -0.98402, 'Me2': 28.0314,
                                     'Me3': 42.0471, 'Myr': 210.1984,
                                     'nitro': 44.98508}

    # Dictionary of monoisotopic weights of Amino Acids in Da, to 5 sf
    peptide_monoisotopic_masses = {'A': 71.03711, 'R': 156.10111, 'N': 114.04293,
                                   'D': 115.02694, 'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
                                   'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
                                   'M': 131.04049, 'F': 147.06841, 'P': 97.05276, 'S': 87.03203, 'T': 101.04768,
                                   'W': 186.07931, 'Y': 163.06333, 'V': 99.06841}
    # Monoisotopic weights of terminal H and OH in Da
    # H = 1.00782
    # OH = 17.00274

    # Split string of sequence
    sequence_list = list(peptide_sequence)

    # Count number of characters in the sequence, equal to or slightly greater than
    # number of amino acids
    num_aa_prelim = len(sequence_list)

    # Declare array for different m/z ratios and fill with row and column headers
    table = np.zeros((num_aa_prelim + 1, 9))

    for num in range(1, num_aa_prelim + 1):
        table[num, 1] = '%.1f' % num

    table[0, 2] = '%.1f' % 2.1
    table[0, 3] = '%.1f' % 2.2
    table[0, 4] = '%.1f' % 2.3

    table[0, 6] = '%.1f' % 25.1
    table[0, 7] = '%.1f' % 25.2
    table[0, 8] = '%.1f' % 25.3

    # Declare array for PTM monoisotopic masses to be referred to making y ion table
    PTM_list = np.zeros((num_aa_prelim))  # of course not this many but it's a
    # small number anyway

    # Calculate m/z values for b(+), b(2+) and b(3+) ions

    # Initialisation
    PTM = ''  # PTM is so far an empty string
    PTM_count = 0
    modified = False
    sequence_overshoot = 0

    b_ion = 1.00782  # monoisotopic mass of H
    y_ion = 17.00274 + 2 * 1.00782  # monoisotopic mass of OH + 2*H

    row_in_table = 1

    for aa in sequence_list:

        if modified == True:

            if aa == ')':
                modified = False
                b_ion = b_ion + PTM_monoisotopic_mass_changes[PTM]

                PTM_list[PTM_count] = PTM_monoisotopic_mass_changes[PTM]  # 1st PTM
                # goes in 0th element, etc.
                PTM = ''  # reset PTM as empty type
                PTM_count = PTM_count + 1

                table[row_in_table, 2] = '%.4f' % b_ion

                b_ion_pp = (b_ion + 1.00782) / 2  # 1.00782 Da is monoistopic mass of
                # H
                table[row_in_table, 3] = '%.4f' % b_ion_pp

                b_ion_ppp = (b_ion + 2 * 1.00782) / 3  # 1.00782 Da is monoistopic mass
                # of H
                table[row_in_table, 4] = '%.4f' % b_ion_ppp

                row_in_table = row_in_table + 1

            else:
                PTM = PTM + aa  # in this case,it's not an amino acid but notation
                # 'aa' retained
                sequence_overshoot = sequence_overshoot + 1

        elif aa == '(':

            row_in_table = row_in_table - 1
            sequence_overshoot = sequence_overshoot + 2  # compensates for first and
            # last bracket
            modified = True
            continue

        else:
            b_ion = b_ion + peptide_monoisotopic_masses[aa]
            table[row_in_table, 2] = '%.4f' % b_ion

            b_ion_pp = (b_ion + 1.00782) / 2  # 1.00782 Da is monoistopic mass of H
            table[row_in_table, 3] = '%.4f' % b_ion_pp

            b_ion_ppp = (b_ion + 2 * 1.00782) / 3  # 1.00782 Da is monoistopic mass of H
            table[row_in_table, 4] = '%.4f' % b_ion_ppp

            row_in_table = row_in_table + 1

    num_aa = num_aa_prelim - sequence_overshoot

    table = table[:(num_aa + 1), :]

    # Now calculate for y ions
    # Different setup entirely here

    row_in_table = 1  # start filling from element 1 in table again
    modified = False  # should already be from loop for b ions
    in_PTM = False  # is string scanning taking place inside a PTM?
    PTM_indicator = PTM_count - 1  # -1 because Python indexing starts on 0 and
    # going to call in from table PTM_list created above

    for aa in reversed(sequence_list):

        if in_PTM == True:

            if aa == '(':
                modified = True
                PTM_to_add = PTM_list[PTM_indicator]
                PTM_indicator = PTM_indicator - 1
                in_PTM = False

            else:
                continue

        elif aa == ')':

            in_PTM = True

        else:

            y_ion = y_ion + peptide_monoisotopic_masses[aa]

            if modified == True:
                y_ion = y_ion + PTM_to_add
                modified = False

            table[row_in_table, 6] = '%.4f' % y_ion

            y_ion_pp = (y_ion + 1.00782) / 2  # 1.00782 Da is monoistopic mass of H
            table[row_in_table, 7] = '%.4f' % y_ion_pp

            y_ion_ppp = (y_ion + 2 * 1.00782) / 3  # 1.00782 Da is monoistopic mass of H
            table[row_in_table, 8] = '%.4f' % y_ion_ppp

            row_in_table = row_in_table + 1

    peptide_mass = b_ion + 17.00274  # monoisotopic mass of OH
    peptide_mass_1 = y_ion - 1.00782  # monoisotopic mass of H
    # these two masses should be identical, we happen to take the mass
    # 'peptide_mass' for the peptide ions m/z

    assert abs(peptide_mass - peptide_mass_1) < 1e-6

    table[0, 0] = '%.4f' % peptide_mass

    for x in range(2, table.shape[0]):
        table[x, 0] = '%.4f' % ((peptide_mass + ((x - 1) * 1.00782)) / float(x - 1))

    return table


def idIons(peakList, xxx_todo_changeme, NLs=['H2O', 'NH3'], fragIonTol=1.5, \
           sortCol=4, printNo=50, mzDev=False, acThresh=5.5, loseAA=2):
    'fragIonTol condition is greater than or equal to. sortCol: 4=sig, 5=vol'
    (sequence, parentZ) = xxx_todo_changeme
    """This filters features according to a/c threshold, then matches all 
    features in the list, then orders the features list according to 
    significance or volume, then prints out the top printNo features."""

    NLMonoisotopicMasses = {'H2O': 18.01056, 'NH3': 17.02653, 'CH3': 14.0157, 'H3PO4': 97.97689, 'HPO3': 79.96633,
                            'SO3': 79.95682, 'None': 0.0}

    colMatchDict = {2: [(6, [2])], 3: [(6, [2, 3]), (7, [2])], \
                    4: [(8, [2]), (7, [2, 3]), (6, [2, 3, 4])]}  # the columns go from
    # low to high charge states at the moment because with histone searches
    # (24/2/2017) we want to match low charge state with low charge state
    # (=> you've had the loss you need to open a unique signal channel)
    # Matches charge state of ion with columns to try to match. First element of
    # tuple is column to look for y ion, second is where b ion might be

    NLs.append('None')  # so that no NL is always considered

    fullNumMatches = 0

    matchedPeaks = []

    table = byPosIons(sequence)
    sequenceLength = table.shape[0] - 1
    ionList = []

    peakList = np.array(peakList)  # so it can be masked on next line
    peakList = peakList[np.array([(abs(peak[0] - peak[1]) > acThresh) \
                                  for peak in peakList])]  # filter out all peaks with m/z's within acThresh

    for peak in peakList:
        ionList.append([peak[0], peak[1], peak[2], peak[3]])  # peak[3] is significance,
        ionList.append([peak[1], peak[0], peak[2], peak[3]])  # peak[2] is volume

    firstOrSecond = 1  # if it's 1 then it's the second time of trying
    passNextTime = False
    for ionPair in ionList:
        firstOrSecond *= -1
        if passNextTime:
            passNextTime = False
            continue
        ion1 = ionPair[0]
        ion2 = ionPair[1]

        goodCorrelation = False
        hits = 0

        ion1Name = 'No'
        ion2Name = 'No'

        ion1mz = 'match'
        ion2mz = 'match'

        for compareCols in colMatchDict[parentZ]:
            yIonAt = compareCols[0]
            yIonCharge = float(yIonAt - 5)
            for NL1 in NLs:
                for row1 in range(1,
                                  sequenceLength):  # not sequenceLength + 1 because you can't have a b1/y(n-1) correlation
                    if abs(ion1 - (table[row1, yIonAt] - NLMonoisotopicMasses[NL1] / yIonCharge)) <= fragIonTol:

                        for bIonAt in compareCols[1]:
                            bIonCharge = float(bIonAt - 1)
                            # For b ions
                            for NL2 in NLs:
                                for numLostAA1 in range(loseAA + 1):
                                    row2 = sequenceLength - row1 - numLostAA1
                                    if abs(ion2 - (table[row2, bIonAt] - NLMonoisotopicMasses[
                                        NL2] / bIonCharge)) <= fragIonTol and row2 > 1:  # row2 > 1 because can't have b1 ions
                                        goodCorrelation = True
                                        hits += 1
                                        if NL2 == 'None':
                                            ion2Name = 'b' + str(row2) + '(' + str(int(bIonCharge)) + '+)'
                                        else:
                                            ion2Name = '[b' + str(row2) + '-' + NL2 + '](' + str(int(bIonCharge)) + '+)'
                                        ion2mz = table[row2, bIonAt] - NLMonoisotopicMasses[
                                            NL2] / bIonCharge  # took the rounding from here 20160427

                            # For a ions
                            for NL2 in NLs:
                                for numLostAA2 in range(loseAA + 1):
                                    row2 = sequenceLength - row1 - numLostAA2
                                    if abs(ion2 - (table[row2, bIonAt] - (NLMonoisotopicMasses[
                                                                              NL2] + 27.99492) / bIonCharge)) <= fragIonTol and row2 > 1:  # 27.99492 is the monoisotopic mass of CO. row2 > 1 because can't have b1 ions
                                        goodCorrelation = True
                                        hits += 1
                                        if NL2 == 'None':
                                            ion2Name = 'a' + str(row2) + '(' + str(int(bIonCharge)) + '+)'
                                        else:
                                            ion2Name = '[a' + str(row2) + '-' + NL2 + '](' + str(int(bIonCharge)) + '+)'
                                        ion2mz = table[row2, bIonAt] - (NLMonoisotopicMasses[
                                                                            NL2] + 27.99492) / bIonCharge  # took the rounding from here 20160427
                                        # changed20160427 (from parentZ to bIonCharge for divisor of NLMosoisotopicMass)
                            if goodCorrelation:
                                fullNumMatches += 1
                                if NL1 == 'None':
                                    ion1Name = 'y' + str(row1) + '(' + str(int(yIonCharge)) + '+)'
                                else:
                                    ion1Name = '[y' + str(row1) + '-' + NL1 + '](' + str(int(yIonCharge)) + '+)'
                                ion1mz = table[row1, yIonAt] - NLMonoisotopicMasses[
                                    NL1] / yIonCharge  # took the rounding from here 20160427
                                # changed20160427 (from parentZ to yIonCharge for divisor of NLMosoisotopicMass)

        appendum = [(ion1Name, ion1mz), ion1, (ion2Name, ion2mz), ion2, round(ionPair[3], 2), round(ionPair[2], 2),
                    goodCorrelation, hits]
        # took the rounding from here 20160427
        if goodCorrelation:
            matchedPeaks.append(appendum)
            if firstOrSecond == -1:
                passNextTime = True
        elif firstOrSecond == 1:
            matchedPeaks.append(appendum)

    # Now order features according to significance or volume
    scores = [match[sortCol] for match in matchedPeaks]
    matchesOrdered = [matchedPeaks[index] for index in reversed(np.argsort(scores))]

    ionsFrom = input('Please provide short description of the experimental data:\n')
    rawDataPath = input('Please provide the full path of the feature list numpy array:\n')

    print('\nExperimental data from: ' + ionsFrom)

    print('Compared with theoretical ions from: ' + sequence + '(' + str(parentZ) + '+)')

    if sortCol == 4:
        print('Sorted by significance')
    elif sortCol == 5:
        print('Sorted by volume')

    print('NLs considered:')
    for NLtoPrint in NLs:
        if NLtoPrint != 'None':
            print(NLtoPrint)

    print('Threshold =', str(fragIonTol))

    print(loseAA, 'consecutive aa losses')

    print('All peaks within', str(acThresh), 'Da of one another discarded')

    for x in range(printNo):

        if matchesOrdered[x][6]:
            print(matchesOrdered[x][0][0], '@', round(matchesOrdered[x][1], 2), '&', matchesOrdered[x][2][0], '@',
                  str(round(matchesOrdered[x][3], 2)) + ':', matchesOrdered[x][sortCol])
            # strange formatting syntax before colon in order to have no space between colon and preceeding number
            # put the rounding here 20160427

        else:
            print(round(matchesOrdered[x][1], 2), '&', str(round(matchesOrdered[x][3], 2)) + ':',
                  matchesOrdered[x][sortCol])
            # strange formatting syntax before colon in order to have no space between colon and preceeding number
            # put the rounding here 20160427

    if mzDev:
        sumForRMS = 0
        countForRMS = 0

        print('\nMass deviations from theoretical:')

        for x in range(printNo):  # run over the list again but this time to do the
            # mass deviation analysis
            if matchesOrdered[x][6]:
                mzDev1 = matchesOrdered[x][0][1] - matchesOrdered[x][1]
                mzDev2 = matchesOrdered[x][2][1] - matchesOrdered[x][3]
                sumForRMS += (mzDev1 ** 2 + mzDev2 ** 2)
                countForRMS += 1

                print('Peak ', x + 1, ' - ', matchesOrdered[x][0][0], ': ', round(mzDev1, 2), ' & ',
                      matchesOrdered[x][2][0], ': ', round(mzDev2, 2))
                # put the rounding here 20160427

        print('Total RMS deviation of ', countForRMS, 'islands identified at tol. ', fragIonTol, ' (', countForRMS * 2,
              ' m/zs)')
        print('= ', (sumForRMS / (countForRMS * 2)) ** 0.5)

    print('\nFull path of numpy array: ' + rawDataPath)

    return matchesOrdered


def readTextFile(textFile, scanFolder, numscanInterval=None):
    """Function reads in mass spectral scan from .txt file of Thermo LTQ XL MS 
    after conversion from .raw file by Xcalibur software (File Converter tool).
    Reads text file line-by-line, it's typically too big to load into RAM.
    
    Void function, performs:
    1) Create scan folder
    2) Save in scan folder 'array' with rows 1: being intensity readings 
    sampled at the m/z specified in row 0.
    3) Save in scan folder 'array_parameters.npy', which holds details of the 
    scan itself and how it was read in. It is an artefact of how the pC2DMS 
    software used to run, and many fields are now irrelevant (many other 
    fields are information that is available elsewhere, e.g. through direct 
    inspection of 'array.npy', but are accessed much faster using 
    'array_parameters.npy'). Details of each field found in
    "D:\Taran\Computing\Software\fullScript\Parameter_File_Indexing.txt.
    
    This software refer to each m/z sampling point as a 'slice', Thermo 
    software refers to it as a 'packet'.
    
    This function assumes uniform spacing in m/z of points at which the
    ion intensity is sampled, and consistent sampling points across all scans. 
    Non-uniform sampling density that is consistent across scans may require 
    interpolation and integration, any sampling density that is
    inconsistent across scans definitely requires interpolation and 
    integration. 
    Code to implement this (which does so as a stand-alone .py program) is 
    found in:
    "D:/Taran/Computing/Software/fullScript/usefulArchive/ReadTextFile.py"
    """

    fullScanFolder = scanFolder  # 'D:/Taran/Data/'+scanFolder, this was changed
    # cause it was useless
    print('Reading file ' + textFile + ' to ' + fullScanFolder)

    if not os.path.exists(fullScanFolder):
        os.mkdir(fullScanFolder)

    startLine = 24  # line tells you number of readings
    lineWithPacket = 36  # line of first packet
    scan = 0  # set to 0, to be increased over iterations
    mzSlice = 0  # set to 0, to be increaed over iterations

    with open(textFile) as f:
        for i, line in enumerate(f):

            if i == 4:  # 5th line, tells you how many scans in text file. This
                # has historically occasionally varied from the number of scans
                # actually read in.
                firstScan = int(line.split(',')[0].split(' ')[2])
                lastScan = int(line.split(',')[1].split(' ')[3])

                numScans = (lastScan - firstScan) + 1

            elif i == 5:  # 6th line, tells you min m/z and max m/z of the scan
                minMZ = float(line.split(',')[0].split(' ')[2])
                maxMZ = float(line.split(',')[1].split(' ')[3])

            if i == startLine:
                numSlices = int(line.split(',')[0].split(' ')[2])  # gives the
                # number of intensity sampling points in this first scan,
                # should be the same number as for the other scans in the file

                array = np.zeros((numScans + 1, numSlices))  # declare final
                # array of all spectra now we have number of scans and
                # number of slices

            if i == lineWithPacket:

                if mzSlice < numSlices:  # indexing for mzSlice starts at 0

                    array[0, mzSlice] = \
                        float("{0:.3f}".format(float(line.split(',')[2]. \
                                                     split(' ')[3])))

                    mzSlice += 1

                    lineWithPacket += 3  # spacing between
                    # lines with info on is 3 lines

                else:
                    break  # break this loop to avoid needing to test both
                    # of the above conditions for each line.

    # Now reset these two
    scan = 0
    mzSlice = 0

    with open(textFile) as f:
        for i, line in enumerate(f):

            if i == startLine:

                scan += 1
                if scan % 100 == 0:
                    print('Reading scan number ' + str(int(scan)))
                mzSlice = 0
                lineWithPacket = startLine + 12
                startLine = startLine + 16 + (3 * numSlices)  # next scan will
                # have a different value for startLine

            elif i == lineWithPacket:
                if mzSlice < numSlices:  # This conditional shouldn't be needed
                    # provided the number of slices is the same in each scan as (is
                    # declared at the start of) the first scan. I have kept it here
                    # because I recall one time I thought this was not the case.
                    try:
                        splitLine = line.split(',')
                        array[scan][mzSlice] = float(splitLine[1].split(' ')[3])
                        mzSlice += 1  # increase mz Slice we're filling
                        lineWithPacket += 3  # spacing between
                        # lines with info on is 3 lines
                    except IndexError:
                        pass

    f.close()

    # Parameters for 'array_parameters.npy'
    params = np.zeros(8)

    params[0] = numSlices
    params[1] = scan# Total number of scans read in to the array. For some
    # reason, historically sometimes not all scans declared at the top of the
    # text file have been read in. It has been found that when this happens,
    # it does not affect the other scans that are read in to the array.
    # The problem might be a result of manual interference with the Python
    # console  as it is running, so advise not to check variables etc. until
    # all is finished.
    params[2] = numScans  # Total number of scans specified at top of text file.
    # See above, historically it has sometimes been the case that this number
    # is larger than the number of scans actually read in.
    sliceSize = np.average(np.diff(array[0, :]))  # Computational limits (i.e.
    # binary representation of numbers) means that some specified slice sizes
    # (e.g. 1/3 Da for Turbo mode in LTQ XL) are represented as e.g. two times
    # 0.33 and one time 0.34 (and repeat), so this step provides uniformly
    # spaced m/z's. CAREFUL here though - if you start needing super precise
    # sub-Da resolution on maps, you will have to omit this step and deal with
    # non-uniform spacings in m/z as they come.
    params[3] = sliceSize
    params[4] = 0  # this was 'interpInt', not needed when no interpolation
    # takes place.
    params[5] = minMZ  # as specified in the header of the full text file
    params[6] = maxMZ  # as specified in the header of the full text file
    params[7] = sliceSize  # this was oneDDataInt which is not needed when no
    # interpolation takes place (and automatically has the same value as
    # sliceSize when there is no interpolation and integration).

    array = array[:scan + 1, :]  # truncates the array on the occasion that the
    # number of scans read in is lower than the number of scans declared at the
    # text file, see above.
    # array = array[:9001]

    if numscanInterval is not None:
        if os.path.basename(fullScanFolder) != str(numScans) + ' scans':
            os.rmdir(fullScanFolder)
            fullScanFolder = os.path.abspath(os.path.join(fullScanFolder, os.pardir)) + '/' + str(numScans) + ' scans'
            if not os.path.exists(fullScanFolder):
                os.makedirs(fullScanFolder)

    np.save(fullScanFolder + '/array.npy', array)
    np.save(fullScanFolder + '/array_parameters.npy', params)

    print('readTextFile complete for ' + fullScanFolder)

    if numscanInterval is not None:
        for numscan in np.arange(numscanInterval, numScans, numscanInterval):
            params[1] = np.array([numscan, scan]).min()
            params[2] = numscan
            fullScanFolder = os.path.abspath(os.path.join(fullScanFolder, os.pardir)) + '/' + str(numscan) + ' scans'
            if not os.path.exists(fullScanFolder):
                os.makedirs(fullScanFolder)
            np.save(fullScanFolder + '/array.npy', array[:numscan + 1])
            np.save(fullScanFolder + '/array_parameters.npy', params)
            print('readTextFile complete for ' + fullScanFolder)

    return

# If you want to do median filtering you can use:
# scipy.ndimage.filters.median_filter(p2.cutAC(map1.array), size=8)
