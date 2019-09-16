#! /usr/bin/python3

import numpy as np
from scipy.io import loadmat
from argparse import ArgumentParser, RawTextHelpFormatter

parser = ArgumentParser(formatter_class=RawTextHelpFormatter,
                        description="Convert MOT experiment data matlab files to pyton format.",
                        epilog="If using ipython3, indicate end of ipython arg parser via '--':\n   $ ipython3 convert_...py -- <args>")

parser.add_argument(dest="matfilename", metavar="datafile.mat", type=str,
                    help="Matlab data file")

args = parser.parse_args()

STRUCT = (
    ("subNum"                                       , np.int64 ),
    ("trial_number_actual"                          , np.int64 ),
    ("trial_number_name"                            , np.int64 ),
    ("speed"                                        , np.float64 ),
    ("condition"                                    , str ),
    ("Accuracy_Percent_Correct"                     , np.float64 ),
    ("Accuracy_NumItems_Correct"                    , np.int64 ),
    ("targets"                                      , np.int64 ),
    ("colors_cluster1_123"                          , np.int64 ),
    ("colors_cluster2_456"                          , np.int64 ),
    ("colors_cluster3_7"                            , np.int64 ),
    ("tracking_duration"                            , np.float64 ),
    ("Choose_response_numbers"                      , np.int64 ),
    ("target_response_numbers"                      , np.int64 ),
    ("Response_1"                                   , np.int64 ),
    ("Response_1_stim"                              , np.int64 ),
    ("Response_2"                                   , np.int64 ),
    ("Response_2_stim"                              , np.int64 ),
    ("Response_3"                                   , np.int64 ),
    ("Response_3_stim"                              , np.int64 ),
    ("response_1_correct"                           , bool ),
    ("response_2_correct"                           , bool ),
    ("response_3_correct"                           , bool ),
    ("Response_1_Time"                              , np.float64 ),
    ("Response_2_Time"                              , np.float64 ),
    ("Response_3_Time"                              , np.float64 ),
    ("Event_1_ItemsOn_Stationary_and_ClustersCued"  , None ),
    ("Event_2_ItemsOn_StartToMove_and_ClustersCued" , None ),
    ("Event_3_ItemsOn_Moving_TargetCuesOn"          , None ),
    ("Event_4_TrackingPeriod_Starts_AllBlackDots"   , None ),
    ("Event_5_TrackingPeriod_Ends_AllBlackDots"     , None ),
    ("Start_Frame_FromOriginal"                     , np.int64 ),
    ("Start_Tracking_Frame"                         , np.int64 ),
    ("End_Frame_FromOriginal"                       , np.int64 ),
    ("Block"                                        , np.int64 )
)

assert ".mat" in args.matfilename, "ERROR: Filename does not end on '.mat'! Correct filename entered?"

print(" > Loadind data from file:", args.matfilename)
X = loadmat(args.matfilename)

Y = dict()
for key, dtype in STRUCT:
    if dtype is None:
        print("   > Skipping entry '%s' by user request (None)." % key)
        continue
    y = np.array([x for x in X['ResponseInfo'][key][0]]).astype(dtype).squeeze()
    print("   > Found entry '%s' of type '%s' with shape = %s" % (key, str(dtype), str(y.shape)))
    Y[key] = y


assert 'STRUCT' is not Y.keys(), "ERROR: Cannot add 'STRUCT' to dictionary (already exists)."
Y["STRUCT"] = STRUCT

print(" > Data loaded successfully.")


outfname = args.matfilename.replace('.mat', '.npz')
print(" > Save data to file '%s'." % outfname)

np.savez_compressed(outfname, **Y)

print(" > Done.")

