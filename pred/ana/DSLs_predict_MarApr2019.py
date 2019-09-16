experiment_label = "prediction_MarApr2019"
conditions = ("GLO", "CLU", "CDH")
subjects = ("00107", "00004", "00285", "00121", "00595", "00188", "00512", "00208", "00007", "00001", "00762", "00311")
perm =[2,9,7,6,0,11,3,10,1,4,5,8]       # Permutation in plots for improved anonymity
subjects = tuple([subjects[i] for i in perm])

DSLs = dict()
for cond in conditions:
    DSLs[cond] = dict()

# # #  PARTICIPANT : 00107
DSLs["GLO"]["00107"] = dict(
    experiment = "2019-03-26-10-47-59-579319_uid_00107_glo",
    kals_noiseless = "2019-04-16-18-42-13-727957_pred_datarun_for_2019-03-26-10-47-59-579319_uid_00107_glo",
    )
DSLs["CLU"]["00107"] = dict(
    experiment = "2019-03-26-11-11-56-627960_uid_00107_clu",
    kals_noiseless = "2019-04-16-18-44-42-345613_pred_datarun_for_2019-03-26-11-11-56-627960_uid_00107_clu",
    )
DSLs["CDH"]["00107"] = dict(
    experiment = "2019-03-26-11-46-08-102422_uid_00107_cdh67",
    kals_noiseless = "2019-04-16-18-45-21-960911_pred_datarun_for_2019-03-26-11-46-08-102422_uid_00107_cdh67",
    )

# # #  PARTICIPANT : 00004
DSLs["GLO"]["00004"] = dict(
    experiment = "2019-03-27-13-40-49-654920_uid_00004_glo",
    kals_noiseless = "2019-04-16-18-46-44-143736_pred_datarun_for_2019-03-27-13-40-49-654920_uid_00004_glo",
    )
DSLs["CDH"]["00004"] = dict(
    experiment = "2019-03-27-14-05-01-154611_uid_00004_cdh67",
    kals_noiseless = "2019-04-16-18-47-27-898351_pred_datarun_for_2019-03-27-14-05-01-154611_uid_00004_cdh67",
    )
DSLs["CLU"]["00004"] = dict(
    experiment = "2019-03-27-14-26-13-678968_uid_00004_clu",
    kals_noiseless = "2019-04-16-18-48-10-966670_pred_datarun_for_2019-03-27-14-26-13-678968_uid_00004_clu",
    )

# #  PARTICIPANT : 00285
DSLs["CDH"]["00285"] = dict(
    experiment = "2019-03-28-14-01-06-757059_uid_00285_cdh67",
    kals_noiseless = "2019-04-16-18-48-51-759010_pred_datarun_for_2019-03-28-14-01-06-757059_uid_00285_cdh67",
    )
DSLs["CLU"]["00285"] = dict(
    experiment = "2019-03-28-14-29-18-163095_uid_00285_clu",
    kals_noiseless = "2019-04-16-18-49-51-189042_pred_datarun_for_2019-03-28-14-29-18-163095_uid_00285_clu",
    )
DSLs["GLO"]["00285"] = dict(
    experiment = "2019-03-28-14-52-27-390105_uid_00285_glo",
    kals_noiseless = "2019-04-16-18-50-55-634743_pred_datarun_for_2019-03-28-14-52-27-390105_uid_00285_glo",
    )


# #  PARTICIPANT : 00121
DSLs["CDH"]["00121"] = dict(
    experiment = "2019-03-28-17-34-32-599340_uid_00121_cdh67",
    kals_noiseless = "2019-04-16-18-51-59-044832_pred_datarun_for_2019-03-28-17-34-32-599340_uid_00121_cdh67",
    )
DSLs["GLO"]["00121"] = dict(
    experiment = "2019-03-28-18-01-08-313354_uid_00121_glo",
    kals_noiseless = "2019-04-16-18-52-24-313104_pred_datarun_for_2019-03-28-18-01-08-313354_uid_00121_glo",
    )
DSLs["CLU"]["00121"] = dict(
    experiment = "2019-03-28-18-33-54-748845_uid_00121_clu",
    kals_noiseless = "2019-04-16-18-52-54-282755_pred_datarun_for_2019-03-28-18-33-54-748845_uid_00121_clu",
    )


# #  PARTICIPANT : 00595
DSLs["CLU"]["00595"] = dict(
    experiment = "2019-03-29-14-08-42-108128_uid_00595_clu",
    kals_noiseless = "2019-04-16-18-53-48-627941_pred_datarun_for_2019-03-29-14-08-42-108128_uid_00595_clu",
    )
DSLs["CDH"]["00595"] = dict(
    experiment = "2019-03-29-14-44-18-442203_uid_00595_cdh67",
    kals_noiseless = "2019-04-16-18-54-17-081694_pred_datarun_for_2019-03-29-14-44-18-442203_uid_00595_cdh67",
    )
DSLs["GLO"]["00595"] = dict(
    experiment = "2019-03-29-15-14-29-985817_uid_00595_glo",
    kals_noiseless = "2019-04-16-18-55-22-356567_pred_datarun_for_2019-03-29-15-14-29-985817_uid_00595_glo",
    )


# #  PARTICIPANT : 00188
DSLs["CLU"]["00188"] = dict(
    experiment = "2019-03-29-16-36-57-717512_uid_00188_clu",
    kals_noiseless = "2019-04-16-18-55-45-648464_pred_datarun_for_2019-03-29-16-36-57-717512_uid_00188_clu",
    )
DSLs["GLO"]["00188"] = dict(
    experiment = "2019-03-29-17-07-28-195936_uid_00188_glo",
    kals_noiseless = "2019-04-16-18-56-15-776176_pred_datarun_for_2019-03-29-17-07-28-195936_uid_00188_glo",
    )
DSLs["CDH"]["00188"] = dict(
    experiment = "2019-03-29-17-37-33-675438_uid_00188_cdh67",
    kals_noiseless = "2019-04-16-18-57-10-324431_pred_datarun_for_2019-03-29-17-37-33-675438_uid_00188_cdh67",
    )



# #  PARTICIPANT : 00512
DSLs["CDH"]["00512"] = dict(
    experiment = "2019-04-02-13-50-23-725625_uid_00512_cdh67",
    kals_noiseless = "2019-04-16-18-58-28-988384_pred_datarun_for_2019-04-02-13-50-23-725625_uid_00512_cdh67",
    )
DSLs["CLU"]["00512"] = dict(
    experiment = "2019-04-02-14-22-11-113426_uid_00512_clu",
    kals_noiseless = "2019-04-16-18-58-59-062124_pred_datarun_for_2019-04-02-14-22-11-113426_uid_00512_clu",
    )
DSLs["GLO"]["00512"] = dict(
    experiment = "2019-04-02-15-09-05-704035_uid_00512_glo",
    kals_noiseless = "2019-04-16-18-59-26-535821_pred_datarun_for_2019-04-02-15-09-05-704035_uid_00512_glo",
    )


# #  PARTICIPANT : 00208
DSLs["CDH"]["00208"] = dict(
    experiment = "2019-04-02-16-27-17-234202_uid_00208_cdh67",
    kals_noiseless = "2019-04-16-19-00-45-153218_pred_datarun_for_2019-04-02-16-27-17-234202_uid_00208_cdh67",
    )
DSLs["GLO"]["00208"] = dict(
    experiment = "2019-04-02-16-58-20-904985_uid_00208_glo",
    kals_noiseless = "2019-04-16-19-01-12-616074_pred_datarun_for_2019-04-02-16-58-20-904985_uid_00208_glo",
    )
DSLs["CLU"]["00208"] = dict(
    experiment = "2019-04-02-17-29-10-727838_uid_00208_clu",
    kals_noiseless = "2019-04-16-19-01-45-249328_pred_datarun_for_2019-04-02-17-29-10-727838_uid_00208_clu",
    )


# #  PARTICIPANT : 00007
DSLs["GLO"]["00007"] = dict(
    experiment = "2019-04-04-10-38-01-110709_uid_00007_glo",
    kals_noiseless = "2019-04-16-19-09-10-513259_pred_datarun_for_2019-04-04-10-38-01-110709_uid_00007_glo",
    )
DSLs["CDH"]["00007"] = dict(
    experiment = "2019-04-04-11-05-42-965850_uid_00007_cdh67",
    kals_noiseless = "2019-04-16-19-09-33-149445_pred_datarun_for_2019-04-04-11-05-42-965850_uid_00007_cdh67",
    )
DSLs["CLU"]["00007"] = dict(
    experiment = "2019-04-04-11-29-17-036401_uid_00007_clu",
    kals_noiseless = "2019-04-16-19-10-12-785865_pred_datarun_for_2019-04-04-11-29-17-036401_uid_00007_clu",
    )


# #  PARTICIPANT : 00001
DSLs["GLO"]["00001"] = dict(
    experiment = "2019-04-05-13-57-15-726308_uid_00001_glo",
    kals_noiseless = "2019-04-16-19-10-58-836479_pred_datarun_for_2019-04-05-13-57-15-726308_uid_00001_glo",
    )
DSLs["CLU"]["00001"] = dict(
    experiment = "2019-04-05-14-20-24-265335_uid_00001_clu",
    kals_noiseless = "2019-04-16-19-11-38-845268_pred_datarun_for_2019-04-05-14-20-24-265335_uid_00001_clu",
    )
DSLs["CDH"]["00001"] = dict(
    experiment = "2019-04-05-14-55-49-928415_uid_00001_cdh67",
    kals_noiseless = "2019-04-16-19-12-21-960756_pred_datarun_for_2019-04-05-14-55-49-928415_uid_00001_cdh67",
    )


# #  PARTICIPANT : 00762
DSLs["CLU"]["00762"] = dict(
    experiment = "2019-04-09-10-36-55-420083_uid_00762_clu",
    kals_noiseless = "2019-04-16-19-13-08-766059_pred_datarun_for_2019-04-09-10-36-55-420083_uid_00762_clu",
    )
DSLs["GLO"]["00762"] = dict(
    experiment = "2019-04-09-11-02-14-081768_uid_00762_glo",
    kals_noiseless = "2019-04-16-19-13-43-678764_pred_datarun_for_2019-04-09-11-02-14-081768_uid_00762_glo",
    )
DSLs["CDH"]["00762"] = dict(
    experiment = "2019-04-09-11-33-10-477474_uid_00762_cdh67",
    kals_noiseless = "2019-04-16-19-14-29-026246_pred_datarun_for_2019-04-09-11-33-10-477474_uid_00762_cdh67",
    )


# #  PARTICIPANT : 00311
DSLs["CLU"]["00311"] = dict(
    experiment = "2019-04-10-13-23-03-429273_uid_00311_clu",
    kals_noiseless = "2019-04-16-19-16-19-900385_pred_datarun_for_2019-04-10-13-23-03-429273_uid_00311_clu",
    )
DSLs["CDH"]["00311"] = dict(
    experiment = "2019-04-10-13-46-01-192560_uid_00311_cdh67",
    kals_noiseless = "2019-04-16-19-16-44-953007_pred_datarun_for_2019-04-10-13-46-01-192560_uid_00311_cdh67",
    )
DSLs["GLO"]["00311"] = dict(
    experiment = "2019-04-10-14-11-02-897819_uid_00311_glo",
    kals_noiseless = "2019-04-16-19-17-19-427231_pred_datarun_for_2019-04-10-14-11-02-897819_uid_00311_glo",
    )



# # #  PARTICIPANT : 00000
# DSLs["GLO"][""] = dict(
#     experiment = "",
#     kals_noiseless = "",
#     )
# DSLs["CLU"][""] = dict(
#     experiment = "",
#     kals_noiseless = "",
#     )
# DSLs["CDH"][""] = dict(
#     experiment = "",
#     kals_noiseless = "",
#     )


