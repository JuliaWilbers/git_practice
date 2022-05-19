# import neccesary packages
from WORC import SimpleWORC
import os

# These packages are only used in analysing the results
import pandas as pd
import json
import fastr
import glob

# If you don't want to use your own data, we use the following example set,
# see also the next code block in this example.
from WORC.exampledata.datadownloader import download_HeadAndNeck

# Define the folder this script is in, so we can easily find the example data
script_path = os.getcwd()

# NOTE: If on Google Colab, uncomment this line
# script_path = os.path.join(script_path, 'WORCTutorial')

# Determine whether you would like to use WORC for binary_classification,
# multiclass_classification or regression
modus = 'binary_classification'

# Download a subset of 20 patients in this folder. You can change these if you want.
nsubjects = 20  # use "all" if you want to download all patients.
data_path = os.path.join(script_path, 'Data')
download_HeadAndNeck(datafolder=data_path, nsubjects=nsubjects)
# Identify our data structure: change the fields below accordingly
# if you use your own data.
imagedatadir = os.path.join(data_path, 'stwstrategyhn1')
image_file_name = 'image.nii.gz'
segmentation_file_name = 'mask.nii.gz'

# File in which the labels (i.e. outcome you want to predict) is stated
# Again, change this accordingly if you use your own data.
label_file = os.path.join(data_path, 'Examplefiles', 'pinfo_HN.csv')

# Name of the label you want to predict
if modus == 'binary_classification':
    # Classification: predict a binary (0 or 1) label
    label_name = ['imaginary_label_1']

elif modus == 'regression':
    # Regression: predict a continuous label
    label_name = ['Age']

elif modus == 'multiclass_classification':
    # Multiclass classification: predict several mutually exclusive binaru labels together
    label_name = ['imaginary_label_1', 'complement_label_1']

# Determine whether we want to do a coarse quick experiment, or a full lengthy
# one. Again, change this accordingly if you use your own data.
coarse = True

# Give your experiment a name
experiment_name = 'Example_STWStrategyHN'

# Instead of the default tempdir, let's but the temporary output in a subfolder
# in the same folder as this script
tmpdir = os.path.join(script_path, 'WORC_' + experiment_name)

# Create a WORC object
experiment = SimpleWORC(experiment_name)

# Set the input data according to the variables we defined earlier
experiment.images_from_this_directory(imagedatadir,
                                      image_file_name=image_file_name)
experiment.segmentations_from_this_directory(imagedatadir,
                                             segmentation_file_name=segmentation_file_name)
experiment.labels_from_this_file(label_file)
experiment.predict_labels(label_name)

# Use the standard workflow for your specific modus
if modus == 'binary_classification':
    experiment.binary_classification(coarse=coarse)
elif modus == 'regression':
    experiment.regression(coarse=coarse)
elif modus == 'multiclass_classification':
    experiment.multiclass_classification(coarse=coarse)

# Set the temporary directory
experiment.set_tmpdir(tmpdir)

# Run the experiment!
experiment.execute()

# Locate output folder
outputfolder = fastr.config.mounts['output']
experiment_folder = os.path.join(outputfolder, 'WORC_' + experiment_name)

print(f"Your output is stored in {experiment_folder}.")

# Read the features for the first patient
# NOTE: we use the glob package for scanning a folder to find specific files
feature_files = glob.glob(os.path.join(experiment_folder,
                                       'Features',
                                       'features_*.hdf5'))
if len(feature_files) == 0:
    raise ValueError('No feature files found: your network has failed.')

feature_files.sort()
featurefile_p1 = feature_files[0]
features_p1 = pd.read_hdf(featurefile_p1)

# Read the overall peformance
performance_file = os.path.join(experiment_folder, 'performance_all_0.json')
if not os.path.exists(performance_file):
    raise ValueError('No performance file found: your network has failed.')
    
with open(performance_file, 'r') as fp:
    performance = json.load(fp)

# Print the feature values and names
print("Feature values from first patient:")
for v, l in zip(features_p1.feature_values, features_p1.feature_labels):
    print(f"\t {l} : {v}.")

# Print the output performance
print("\n Performance:")
stats = performance['Statistics']
for k, v in stats.items():
    print(f"\t {k} {v}.")


