import os
import sys

sys.path.append(os.getcwd())

from database import DB
from utils import test


@test
def test_DB():
    db = DB(r'database/DB.npz')
    feat, table = db.database()
    print(f'feat: {feat.shape} table: {table.shape} len: {len(db)}')

    for adr in table[:10]:
        print(f'---{adr}')


if __name__ == '__main__':
    test_DB()
