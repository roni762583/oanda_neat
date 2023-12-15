
import multiprocessing
from random import random
from multiprocessing import Process
from multiprocessing import Queue
import pandas as pd
import numpy as np
import neat
import pickle
import os
from datetime import datetime, timedelta
import visualize

import wget
import math
import requests
from neat.nn import FeedForwardNetwork
from neat.population import Population
import matplotlib.pyplot as plt
import json
import time
from time import sleep
from src.functions import *
from src.trading_environment import *
from config.acct_config import *
from config.experiment_config import * # instrument, n, initial_balance
import multiprocessing
import neat
import logging
from src.trading_environment import TradingEnvironment
from src.neat_agent import NEATAgent



def get_pip_location():
    ins_df = pd.read_pickle("data/instruments.pkl")
    mypiplocation = ins_df['pipLocation'].iloc[-1]
    print('pip loc: ', mypiplocation)
    return mypiplocation


def preprocess_data():
    # Check if necessary files exist
    files_exist = all(
        [
            os.path.exists("data/train_df.pkl"),
            os.path.exists("data/test_df.pkl"),
            os.path.exists("data/small_train_df.pkl")
        ]
    )

    if not files_exist:
        # Data preprocessing steps...
        

        # Replace 'URL' with the actual URL of the CSV file
        url = 'https://raw.githubusercontent.com/roni762583/AUDJPY-1M/main/DAT_ASCII_AUDJPY_M1_2020.csv'

        # Read the CSV file with ';' as the delimiter
        df = pd.read_csv(url, sep=';', header=None, names=['time', 'Open', 'High', 'Low', 'bid_c','vol'], parse_dates=True)
        df = df.drop(columns=['Open', 'High', 'Low', 'vol'])

        # Convert the 'time' column to pandas datetime if it's not already in datetime format
        df['time'] = pd.to_datetime(df['time'])

        # Calculate the time differences (time deltas)
        time_diffs = df['time'].diff()

        # number of bars for trend line
        trendline_shift = 12
        
        den = math.pow(10, mypiplocation)

        df['shifted_bid'] = df['bid_c'].shift(trendline_shift, axis=0)

        # change since last bar
        df['pips_delta'] = df['bid_c'].diff()/den # az * from /

        # trendline delta tld
        df['tl_delta'] = (df['bid_c'] - df['shifted_bid'])/den

        # add case switch for other granularities
        minutes = 1 #if (granularity == "M1") else (5 if (granularity == "M5") else 1)

        # in pips per minute
        df['tl_slope'] = df['tl_delta']/(minutes*trendline_shift)

        # get rid of extra columns
        df = df.drop(columns=['shifted_bid', 'tl_delta'])

        print(df.shape, ' b4 drop na')

        # Drop rows that contain NaN/None ONLY in columns: 'pips_delta', 'tl_slope'
        df.dropna(subset=['pips_delta', 'tl_slope'], inplace=True)

        print(df.shape, ' after drop na')

        # Reset the index of df
        df = df.reset_index(drop=True)

        # Find the split index (70% of the total rows)
        split_index = int(0.7 * df.shape[0])
        small_split = int(0.0025 * df.shape[0])
        # Create neat_df as the first 70% of rows of data from df
        small_train_df = df.iloc[:small_split]
        train_df = df.iloc[:split_index]
        # Create neat_df as the first 70% of rows of data from df
        test_df = df.iloc[split_index:]

        print('train_df.shape: ', train_df.shape)
        print('small_train_df.shape: ', small_train_df.shape)
        print('test_df.shape: ', test_df.shape)

        # Save data to files
        train_df.to_pickle("data/train_df.pkl")
        test_df.to_pickle("data/test_df.pkl")
        small_train_df.to_pickle("data/small_train_df.pkl")
    mypiplocation = get_pip_location()
    # Load neat_df
    neat_df = pd.read_pickle("data/small_train_df.pkl")
    print('neat_df.shape: ', neat_df.shape)
    return mypiplocation

def create_neat_config():
    # Read the NEAT configuration file and create config object
    config_file = 'config/neat_config.txt'
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    return config

def load_population(config):
    # Create the NEAT population using the provided configuration
    p = neat.Population(config)
    # Create a directory to store checkpoints if it doesn't exist
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Set up the checkpointer to save checkpoints in the specified directory
    checkpoint_interval = 1  # Adjust the checkpoint interval as needed
    checkpointer = neat.Checkpointer(generation_interval=checkpoint_interval,
                                     filename_prefix=os.path.join(checkpoint_dir, 'neat-checkpoint-'))
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(checkpointer) # neat.Checkpointer(1))
    return p

def pickle_best_genome(winner):
    # Save the best-performing genome to a file
    best_genome_filename = "models/best_genome.pkl"
    with open(best_genome_filename, "wb") as best_genome_file:
        pickle.dump(winner, best_genome_file)
    return best_genome_filename

def identify_winner_trades(trade_lists, winner):
    # Identify and save winner agent's trades
    winner_agent_key = winner.key
    winner_agent_trades = None
    # Logic to find winner's trades within trade_lists...
    if winner_agent_trades is not None:
        winner_agent_trade_filename = "trades/winner_agent_trades.txt"
        # Save the winner agent's trades to a file
        with open(winner_agent_trade_filename, "w") as trade_file:
            for trade in winner_agent_trades:
                trade_file.write(trade + '\n')
        logging.info(f"Winner agent's trades saved to {winner_agent_trade_filename}")
    else:
        logging.warning(f"No trade data found for the winner agent with key {winner_agent_key}")


def set_start_method(use_multiprocessing):
    if use_multiprocessing:
        multiprocessing.set_start_method('fork')  # 'spawn'


def eval_genomes(genomes, config):
    #global neat_df
    #global mypiplocation
    nets = []
    agents = []
    ge = []
    total_rewards_list = []
    # create the shared queue
    queue = Queue()
    debuggingqueue = Queue()
    processes = []
    # build agents
    for genome_id, genome in genomes:
        genome.fitness = 0
        ge.append(genome)
        trading_env = TradingEnvironment(neat_df, initial_balance, mypiplocation, None)
        agent = NEATAgent(trading_env, genome, config)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        trading_env.net = net
        agent.set_network(net)
        agents.append(agent) 
        ####################        
        
        # Start a process for each agent to run simulate_trading method
        agent_process = Process(target=agent.simulate_trading, args=(queue,debuggingqueue))
        agent_process.start()
        processes.append(agent_process)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Read queue to get total rewards
    total_rewards_list = []
    while not queue.empty():
        tr = queue.get()
        #print('tr=',tr)
        total_rewards_list.append(tr)
    '''
    print('DEBUGGING QUEUE:')
    while not debuggingqueue.empty():
        print(debuggingqueue.get())
    '''
    # Assign fitness to genomes based on total rewards
    for genome, total_reward in zip(ge, total_rewards_list):
        genome.fitness = total_reward
    '''
    # loop over genomes to assign fitness
    for genome, total_reward in zip(ge, total_rewards_list):
        genome.fitness = total_reward
        print('total_reward======', total_reward)
    '''

def main(use_multiprocessing=True):
    if use_multiprocessing:
        set_start_method(use_multiprocessing)

    get_and_pickle_instrument_info(API_KEY, ACCOUNT_ID, instrument)
    mypiplocation = preprocess_data()
    # print('myPip=', mypiplocation)
    config = create_neat_config()

    population = load_population(config)

    n = 5  # Number of generations to run
    winner = population.run(eval_genomes, n)
    #print('winner is ... ', winner)
    best_genome_filename = pickle_best_genome(winner)

    # Call functions to identify winner's trades within trade_lists
    identify_winner_trades(trade_lists, winner)

    # Display final stats
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    trade_lists = []  # Define trade_lists here or load from somewhere
    mypiplocation = np.nan
    neat_df = pd.DataFrame()
    # Execute the main function
    main(use_multiprocessing=False)
    
    
