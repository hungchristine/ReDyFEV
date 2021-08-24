# -*- coding: utf-8 -*-
from peewee import (
    Model,
    SqliteDatabase,
    TextField,
    IntegerField,
)
import os


_db = SqliteDatabase(None)


class File(Model):
    filename = TextField()
    country = TextField()
    year = IntegerField()
    sha256 = TextField(unique=True)
    kind = TextField()

    class Meta:
        database = _db


def get_database(dirpath):
    _db.init(os.path.join(dirpath, "bentso_cache.db"))
    _db.create_tables([File], safe=True)
    return _db
