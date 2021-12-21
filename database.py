import numpy as np


class DB(object):
    def __init__(self, db_path='database/DB.npz'):
        self.path = db_path
        self._read_db(db_path)

    def _read_db(self, path):
        db = np.load(path)

        self.table = db['INDEX']
        self.feat = db['DATA']

    def __len__(self):
        return len(self.table)

    def database(self):
        return self.feat, self.table
