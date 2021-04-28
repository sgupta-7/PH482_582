"""
- Modifies the data into mutliple formats.
- Oversamples, undersamples, and other data formatting options.
- script also subsets the data

RUN:
- comment out the features in reqd_features that you do not need
- run the script using:
	python3 data_modifer.py
"""

### imports ==================================================================
import random							# create random subset(s)
from pprint import pprint as pprint		# printing
from tqdm import tqdm as tqdm			# progress
import sys								# script creation/testing
import time 							# for scheduled sleeps

### variables ==================================================================

data_directory = "/Users/rele.c/Box/00_Personal/UA_School/03_Spring2021/PH582-Machine_Learning/code/PH582-ML-Final_Project/final/data"

standard_file = "{0}/{1}".format( data_directory, "standard_data.csv" )
standard_subset_file = "{0}/{1}".format( data_directory, "standard_subset.csv" )
oversample_file = "{0}/{1}".format( data_directory, "oversample.csv" )
undersample_file = "{0}/{1}".format( data_directory, "undersample.csv" )

### functions ==================================================================

# to seed further runs
random.seed( 314159265 )

def subset_creator( i="input_file", o="output_file", features="reqd_features", mode="none", iteration=None ):
	"""
	- creates a subset of the input file and outputs it
	- requries reqdfeatures and mode.
	- mode can be:
		- none: no modification
		- oversample: oversamples the file so that B and M have the same count
		- undersample: undersamples the file so that B and M have the same count
	"""

	if iteration != None:
		o = o.split(".csv")[0] + "_{0}.csv".format(iteration)

	from tqdm import tqdm as tqdm			# progress
	import sys								# script creation/testing
	import time 							# for scheduled sleeps
	import random							# create random subset(s)

	if features == "reqd_features":
		sys.exit( "Feature dictionary not added." )

	with open( i, "r" ) as infile, open( o, "w" ) as outfile:

		out_data = []

		for line in tqdm( infile, ascii=True, desc="Reading {0}".format( i.split("/")[-1] ) ):
			lst_line = line.strip().split(",")
			if "diagnosis" in line:
				lst_line = [ x[1:-1] for x in line.strip().split(",") ]

			out_lst = []
			for key in features.keys():
				out_lst.append( lst_line[ features[key] ] )

			out_data.append(  ",".join( out_lst ) )

		if mode == "none":
			for line in out_data:
				print( line, file=outfile )
			return

		if mode == "oversample":
			m_count, b_count = find_least_most( out_data, features["diagnosis"] )
			reqd_samples = b_count - m_count

			# print( out_data[1:] )

			out_data.extend( random.choices( out_data[1:], k=reqd_samples ) )

			for line in out_data:
				print( line, file=outfile )
			return

		if mode == "undersample":
			m_count, b_count = find_least_most( out_data, features["diagnosis"] )

			header = out_data[0]
			m_data, b_data = [], []
			for item in out_data[1:]:
				if item.split(",")[1] == "M":
					m_data.append( item )
				if item.split(",")[1] == "B":
					b_data.append( item )

			random.shuffle( b_data )
			final_out = []
			temp_out = []
			final_out.append( header )
			temp_out.extend( m_data )
			temp_out.extend( b_data[ :len(m_data) ] )

			random.shuffle( temp_out )
			final_out.extend( temp_out )

			for line in final_out:
				print( line, file=outfile )
			return

def find_least_most( lol, index ):
	"""
	- finds the number of the min, and the max of a column from a list of lists
	- returns ( m_count, b_count )
	"""

	column = []
	for line in lol:
		if "diagnosis" in line:
			continue
		column.append( line.split(",")[ index ] )

	return( column.count( "M" ), column.count( "B" ) )

### subset(s) ==================================================================
# creates a subset based on the values that are not commented out in the following list
# those that we do not want in the subset can be commented out.

reqd_features = {
	"id"						:	0	,
	"diagnosis"					:	1	,
	"radius_mean"				:	2	,
	"texture_mean"				:	3	,
	"perimeter_mean"			:	4	,
	"area_mean"					:	5	,
	"smoothness_mean"			:	6	,
	"compactness_mean"			:	7	,
	"concavity_mean"			:	8	,
	"concave points_mean"		:	9	,
	"symmetry_mean"				:	10	,
	"fractal_dimension_mean"	:	11	,
	"radius_se"					:	12	,
	"texture_se"				:	13	,
	"perimeter_se"				:	14	,
	"area_se"					:	15	,
	"smoothness_se"				:	16	,
	"compactness_se"			:	17	,
	"concavity_se"				:	18	,
	"concave points_se"			:	19	,
	"symmetry_se"				:	20	,
	"fractal_dimension_se"		:	21	,
	"radius_worst"				:	22	,
	"texture_worst"				:	23	,
	"perimeter_worst"			:	24	,
	"area_worst"				:	25	,
	"smoothness_worst"			:	26	,
	"compactness_worst"			:	27	,
	"concavity_worst"			:	28	,
	"concave points_worst"		:	29	,
	"symmetry_worst"			:	30	,
	"fractal_dimension_worst"	:	31	,
}

### code ==================================================================

# create standard subset
for i in range( 5 ):
	# subset_creator( i=standard_file, o=standard_subset_file, features=reqd_features, mode="none", iteration=(i+1) )
	subset_creator( i=standard_file, o=oversample_file, features=reqd_features, mode="oversample", iteration=(i+1) )
	subset_creator( i=standard_file, o=undersample_file, features=reqd_features, mode="undersample", iteration=(i+1) )
