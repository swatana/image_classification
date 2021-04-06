import os

class NotFoundError(Exception):
    pass

def get_unused_dir_num(dir_name, pref):
    test_data_path = os.path.join(os.path.dirname(__file__), dir_name)
    os.makedirs(test_data_path, exist_ok=True)
    dir_list = os.listdir(path=test_data_path)
    for i in range(1000):
        search_dir_name = pref + "_" + '%03d' % i
        if search_dir_name not in dir_list:
            return search_dir_name
    raise NotFoundError('Error')