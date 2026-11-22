
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import fastf1 as f1
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from kagglehub import KaggleDatasetAdapter 

#################### Function to select best time from quali results ###############

def best_time(row):
    times = row[["Q1", "Q2", "Q3"]].dropna()
    return times.min().total_seconds() if len(times) else None


################ open excel spreadsheet ####################
# copied from claude, so we'll see how this one goes
# wow worked well.

def print_to_excel(csv_path, data_frame):
    #csv_path = "/home/edward_c_holder/ml/ml_proj_2/output.csv"
    data_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Convert WSL path to Windows path and open it
    #subprocess effectively allows python to act as a terminal
    windows_path = subprocess.run(["wslpath", "-w", csv_path], capture_output=True, text=True).stdout.strip()
    subprocess.run(["explorer.exe", windows_path])
    return

############################################################

if __name__ == "__main__":

    #setup cache

    Path("./f1_cache").mkdir(exist_ok=True)
    f1.Cache.enable_cache("./f1_cache")

    #location for excel path
    csv_path = "/home/edward_c_holder/ml/ml_proj_2/data/output.csv"

    ############## change pandas options to show whole dataframe '####################
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    ######################################################################


    seasons = [2022,2023,2024,2025]


    ######################### loop through each season and event to create full dataframe ############################

    all_races = []
    for season in range(len(seasons)):

        schedule = f1.get_event_schedule(seasons[season])
        rounds = schedule[schedule["RoundNumber"] != 0]
        #print(rounds[["RoundNumber","Location"]])

        for round_index in range(len(rounds["RoundNumber"])):

            race = f1.get_session(seasons[season],rounds["RoundNumber"].iloc[round_index],'R')
            race.load(laps=False,telemetry=False,weather=False,messages=False)
            race_results = race.results[["Abbreviation","TeamName","GridPosition","Position","Status","Points"]].copy()

            quali = f1.get_session(seasons[season],rounds["RoundNumber"].iloc[round_index],'Q')
            quali.load(laps=False,telemetry=False,weather=False,messages=False)
            quali_results = quali.results[["Abbreviation","Q1","Q2","Q3"]].copy()

            #for a binary classifier, generate a podium column now based on position
            race_results["Podium"] = (race_results["Position"] <= 3).astype(float)

            #take best qualifying time from each racer to eliminate issue of having three columns of different length
            quali_results["QualiBestSeconds"] = quali_results.apply(best_time,axis=1)
            pole_time = quali_results["QualiBestSeconds"].min()
            quali_results["QualiGapToPole"] = quali_results["QualiBestSeconds"] - pole_time

            #merge race and qualifying results into one dataframe
            event_data_merged = race_results.merge(
                quali_results[["Abbreviation","QualiBestSeconds","QualiGapToPole"]],
                left_on= "Abbreviation",
                right_on="Abbreviation",
                how="left",
            )

            event_data_merged["Season"] = seasons[season]
            event_data_merged["Round"] = rounds["RoundNumber"].iloc[round_index]
            event_data_merged["Location"] = rounds["Location"].iloc[round_index]
            all_races.append(event_data_merged)
            
            #pause between each round to prevent exceeding cache limit
            time.sleep(1)

    full_dataset = pd.DataFrame(pd.concat(all_races,ignore_index=True))
    #print(full_dataset["Location"].unique())


    # ensure historic team name changes don't chnage which team is in data:
    TEAM_LINEAGE = {"AlphaTauri": "Racing Bulls", "RB": "Racing Bulls",
                "Alfa Romeo": "Kick Sauber"}
    full_dataset["TeamName"] = full_dataset["TeamName"].replace(TEAM_LINEAGE)

    #sort dataset into chronological order (not sure if this was already done?)
    full_dataset = full_dataset.sort_values(["Season", "Round"]).reset_index(drop=True)

    #save dataframe to csv so it doesn't have to be reloaded each time
    full_dataset.to_csv("/home/edward_c_holder/ml/ml_proj_2/data/full_dataset_raw.csv")

    ##############################################################################################

    #reload dataframe

    full_dataset = pd.read_csv("/home/edward_c_holder/ml/ml_proj_2/data/full_dataset_raw.csv")

    #create a rolling race average column

    #first group full dataset by driver and have his position values in order
    #i.e groupby stops dataleakage across drivers
    #x.shift moves the position data down by one, effectively creating a 'last race position column'
    # .rolling() creates a window of the previous 5 values at each row and . mean averages these
    # min_periods parameter says if there's fewer than 5 previous races available it just computes a mean from what's available
    # .transform maps computed groupby values back to their original position


    full_dataset["DriverRollingAvgFinish"] = (
        full_dataset.groupby("Abbreviation")["Position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    full_dataset["DriverTrackHistory"] = (
        full_dataset.groupby(["Abbreviation","Location"])["Position"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )


    ####################calculate average team race result############################
    team_race_avg = full_dataset.groupby(["TeamName","Season","Round"])["Position"].mean().reset_index()
    team_race_avg = team_race_avg.rename(columns={"Position": "TeamAvgPosition"})

    team_race_avg["ConstructorRollingAverage"] = (
        team_race_avg.groupby("TeamName")["TeamAvgPosition"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    full_dataset = full_dataset.merge(
        team_race_avg[["TeamName","Season","Round","ConstructorRollingAverage"]],
        on=["TeamName","Season","Round"],
        how="left"
    )

    ###############################################################################

    # full_dataset["ConstructorRollingAverage"] = (
    #     full_dataset.groupby("TeamName")["Position"]
    #     .transform(lambda x: x.shift(1).rolling(5,min_periods=1).mean())
    # )

    full_dataset["ChampionshipStanding"] = (
        full_dataset.groupby(["Abbreviation","Season"])["Points"]
        .transform(lambda x: x.shift(1).expanding().sum())
    )

    print(full_dataset.columns.tolist())

    full_dataset.to_csv("/home/edward_c_holder/ml/ml_proj_2/data/full_dataset_enriched.csv")

    full_dataset = pd.read_csv("/home/edward_c_holder/ml/ml_proj_2/data/full_dataset_enriched.csv")

    print_to_excel(csv_path,full_dataset)


