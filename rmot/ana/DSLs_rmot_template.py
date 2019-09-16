title = "MOT MyExp"

exppath = "./data/myexp/responses/"
simpath = "./data/sim/"

subjects = range(1,21)
conditions = ("independent_test", "global", "counter", "hierarchy_124", "hierarchy_127")
trackers = ("TRU", "IND")  # Kalman filters to run (TRU will match ground truth)

DSL = {s : dict(exp=None, sim=None) for s in subjects}
for s in subjects:
    DSL[s]["sim"] = {c : None for c in conditions}


# # #  PARTICIPANT N  # # #
# Replace <N> by the participant's number.
# Enter the speed (as float X.XX)
# For each sim, enter the DSL
d = DSL[<N>]
d["speed"] = <0.00>
d["exp"] = "Response_File_Test_P<N>.npz"
d["sim"]["independent_test"] = ""
d["sim"]["global"] = ""
d["sim"]["counter"] = ""
d["sim"]["hierarchy_124"] = ""
d["sim"]["hierarchy_127"] = ""

