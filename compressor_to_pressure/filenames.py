# filenames.py
import glob
import os


nov_folder = "/DatenNovember23"
comp_data_folder = "kompressordaten"
air_leader_folder = "/AirLeader"
flow_folder = "/VolumenstromDruckluft"

compressor_list_file = comp_data_folder + "/compressor_list.csv"
# ===========================================================
air_leader_path = comp_data_folder + nov_folder + air_leader_folder

short_air_leader_file = air_leader_path + "/20231101.LOG.short.csv"
d1_air_leader_file = air_leader_path + "/20231101.LOG.csv"

all_air_leader_files = glob.glob(os.path.join(air_leader_path, "*LOG.csv"))
# ===========================================================
flow_path = comp_data_folder + nov_folder + flow_folder

flow_file = flow_path + "/DurchflussdatenNetzABC_01.11-20.11.23.csv"
short_flow_file = flow_path + "/DurchflussdatenNetzABC_01.11-20.11.23.short.csv"
d1_flow_file = flow_path + "/DurchflussdatenNetzABC_01.11-20.11.23.1day.csv"
# ===========================================================
if __name__ == "__main__":
    print(d1_air_leader_file)
    # print(all_air_leader_files_test)
    print("filenames.py")
