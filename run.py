# -*-coding:utf-8 -*-
# @Author: xiaolizhang

import yaml
import argparse

from utils.data_builder import preprocessing_data
from utils.config import Config


def run():
    conf = Config()

    # build dataset
    dataset = preprocessing_data(conf=conf)



    return None


if __name__ == '__main__':
    run()
