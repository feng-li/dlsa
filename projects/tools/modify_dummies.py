#! /usr/bin/env python3

dummy_info_path = "~/running/data_raw/dummy_info.pkl"
dummy_info_path_latest = "~/running/data_raw/dummy_info_latest.pkl"

with open(os.path.expanduser(dummy_info_path), "rb") as f:
    dummy_info = pickle.load(f)

dummy_info['factor_set']["Year"] = list(range(1987, 2009))
dummy_info['factor_selected']["Year"] = list(range(1987, 2009))
dummy_info['factor_dropped']["Year"] = []
dummy_info['factor_selected_names']["Year"] = [
    "Year" + '_' + str(x) for x in dummy_info['factor_selected']["Year"]
]

pickle.dump(dummy_info, open(os.path.expanduser(dummy_info_path_latest), 'wb'))
print("dummy_info saved in:\t" + dummy_info_path_latest)
