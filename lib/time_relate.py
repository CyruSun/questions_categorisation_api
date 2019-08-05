# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:52:51 2019

@author: Mahery
"""

import pandas as pd
import time

from functools import wraps


"""Module de manipulation et de conversion de temps."""

MINUTES_DAY = 60 * 24


def round_datetime_hour(dt_tm: pd._libs.tslibs.timedeltas.Timedelta):
    """Rounded hour of a datetime.
    Keywords arguments:
        dt_tm -- date de type Timedelta"""
    return round(dt_tm.seconds / 3600)


def elapsed_round_minute(
        date_ref, dt_tm_stmp: pd._libs.tslibs.timestamps.Timestamp):
    """Returns the duration in minutes from a reference date.
    Keywords arguments:
        date_ref -- reference date
        dt_tm_stmp -- date that needs to be compared to the reference date
        (type Timestamp)"""

    # Difference of days converted to minutes
    diff_date = date_ref - dt_tm_stmp  # type: pandas._libs.tslibs.timedeltas
    # .Timedelta
    diff_mnts = diff_date.days * MINUTES_DAY
    # Hours to minutes conversion and rounding
    mnts = diff_date.seconds / 60  # minutes from hours
    rnd_mnts = round(mnts)
    # Recencies
    elpsd_tm = diff_mnts + rnd_mnts
    return elpsd_tm


def elapsed_round_days(
        date_ref, dt_tm_stmp: pd._libs.tslibs.timestamps.Timestamp):
    """Returns the duration from a reference date.
    Keywords arguments:
        date_ref -- reference date
        dt_tm_stmp -- date that needs to be compared to the reference date
        (type Timestamp)"""

    # Difference of days converted to minutes
    diff_date = date_ref - dt_tm_stmp  # type: pandas._libs.tslibs.timedeltas
    # .Timedelta
    return diff_date.days


def timer(func):
    """Closure fonction that times an execution used as decorator."""

    @wraps(func)  # Decorator that enables func to keep its own metadata
    def wrapper(*args, **dargs):
        start = time.time()
        rslt = func(*args, **dargs)
        print('--- {:2} seconds ---'.format(time.time() - start))
        return rslt
    return wrapper
