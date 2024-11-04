# this will incorporate the training loop and testing
# 1 yr of M1 data
# feature extraction
# sharded into days dfs
# exclude days w/ gaps > 1 Hr.
# list of daily df's - name is day year
# check if saved genome (not checkpoint to allow changing config), if yes start training based on genome and config
# loop over df's list, grab 2 at a time i, i+1 :
#    i is used for training n generations - record winner genome
#    i+1 is used for testing winner of training i - record stats: PnL, expectancy, max dd, ..., graph test run w/ trades
# end of daily df's
# add web GUI - training monitoring, live rolling stats, graphs, control button
from src.evaluate_genome_new import * 
import os
import re
import glob
import os.path
import time
from time import sleep
from datetime import datetime, timedelta
import math
import random
from random import random, randint
import multiprocessing
from multiprocessing import Process, Queue, Manager
from multiprocessing.queues import Empty 
import logging
import pandas as pd
import numpy as np
import neat
from neat.nn import FeedForwardNetwork
from neat.population import Population
import copy
import pickle
import platform
import matplotlib.pyplot as plt
import wget
import requests
import json
import config.acct_config
from config.experiment_config import *
from queue import Empty
import gzip
from config.acct_config import *


# new_main.py



class NEATEvaluator:
    def __init__(self):
        pass
        # Initialize anything needed for NEAT evaluation

    def evaluate_genome_wrapper(self, queue, data_tuple, local_simulation_vars):
        return egn.evaluate_genome(queue, data_tuple, local_simulation_vars)

    def eval_gemones_multi(self, genomes, config, simulation_vars, daily_df, queue):
        # Your eval_gemones_multi logic here
        pass

    def main(self):
        # Your main logic here
        pass

if __name__ == "__main__":
    evaluator = NEATEvaluator()
    evaluator.main()



def set_start_method():
    # 'forkserver' # לא עובד
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn')
        print('windows detected - using spawn multiprocessing method')
    else:
        multiprocessing.set_start_method('fork')
        print('non-windows detected - using fork multiprocessing method')
trade_lists = []
daily_df = pd.DataFrame()
# Configure logging at the script level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# set star method for multiprocessing
set_start_method()
# Using multiprocessing Pool to limit the number of processes
num_processes = multiprocessing.cpu_count() #// 2  # Limit to half the CPU cores
pool = multiprocessing.Pool(processes=num_processes)
# Create a Manager object
manager = Manager()
queue = manager.Queue()
global_vars = manager.dict()
top_genomes = manager.list()
gen_num = manager.Value('i', 0)
#setup_global_vars
simulation_vars = {
    'volume': 0,
    'current_step': 0,
    'done': False,
    'balance': initial_balance, # check it is set from config file
    'pip_location': np.nan,
    'cost': -1 * spread,
    'current_timestamp': pd.NaT,
    'current_price': 0.0,
    'open_time': pd.NaT,
    'open_price': 0.0,
    'p_l': -1 * spread,  # Assuming p_l starts with cost
    'close_time': pd.NaT,
    'close_price': 0.0,
    'duration': pd.Timedelta(0),
    'total_ticks': 0,
    'ticks_underwater': 0,
    'ticks_abovewater': 0,
    'underwater_fraction': 0.0,
    'above_water_fraction': 0.0,
    'hwm': -1 * spread,  # Assuming hwm starts with cost
    'lwm': -1 * spread,  # Assuming lwm starts with cost
    'mdd': -1 * spread,  # Assuming mdd starts with cost
    'profit_over_underwater_fraction': 0.0,
    'profit_over_mdd': 0,
    'profit_over_underwater_fraction_and_mdd': 0,
    'equity_curve': [-1 * spread],  # Assuming equity_curve starts with cost
    'direction': 0,  # -1 short, +1 long
    'top_genomes': top_genomes,
    'gen_num': gen_num
}
# Add top_genomes to simulation_vars
simulation_vars['top_genomes'] = top_genomes
simulation_vars['gen_num'] = gen_num

# gets instruments info
def fetch_instrument_data(api_key, account_id, instrument_name):
    # Base OANDA API URL
    oanda_url = "https://api-fxtrade.oanda.com/v3"

    # Endpoint for instrument information
    inst_url = f"{oanda_url}/accounts/{account_id}/instruments"

    # Request headers with authorization
    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    # Create a session for making requests
    session = requests.Session()

    # Make the API request to get instrument information
    inst_response = session.get(inst_url, params=None, headers=headers)

    # Check if the request was successful
    if inst_response.status_code == 200:
        # Extract instrument data from the JSON response
        inst_data = inst_response.json()

        # Extract information for the specified instrument
        instrument_info = next((item for item in inst_data['instruments'] if item['name'] == instrument_name), None)

        if instrument_info:
            # Return a dictionary with instrument information
            return {
                'name': instrument_info['name'],
                'type': instrument_info['type'],
                'displayName': instrument_info['displayName'],
                'pipLocation': instrument_info['pipLocation'],
                'marginRate': instrument_info['marginRate']
            }
        else:
            print(f"Instrument '{instrument_name}' not found.")
            return None
    else:
        print(f"Error: {inst_response.status_code}, {inst_response.text}")
        return None
    
def get_and_pickle_instrument_info(API_KEY, ACCOUNT_ID, instrument):
    # Check if 'instruments.pkl' file exists
    if not os.path.isfile("data/instruments.pkl"):
        # Fetch instrument info
        instrument_info = fetch_instrument_data(API_KEY, ACCOUNT_ID, instrument)

        if instrument_info:
            print("Instrument Info:")
            print(instrument_info)
            # Convert the instrument_info to a list before creating the DataFrame
            instrument_info_list = [instrument_info]
            instrument_df = pd.DataFrame(instrument_info_list)

            # Pickle Instruments data df
            instrument_df.to_pickle("data/instruments.pkl")
    else:
        print("'instruments.pkl' file already exists.")

# to clean observation from bad values
def deNaN(observation):
    #print('hello from deNaN()')
    for i in range(len(observation)):
        if observation[i] is None or np.isnan(observation[i]):
            # Replace NaN values with appropriate default values
            if i == 0:
                observation[i] = 0.0
                # print('pips_delta error for NaN at idx:', self.current_step)
            elif i == 1:
                observation[i] = 0.0
                # print('tl_slope error for NaN at idx:', self.current_step)
            elif i == 2:
                observation[i] = 0.0
                # print('unrealized error for NaN at idx:', self.current_step)
            elif i == 3:
                observation[i] = 0.0
                # print('holding error for NaN at idx:', self.current_step)
    return observation

def calculate_drawdown(simulation_vars):
    peak = max(simulation_vars['equity_curve']) if max(simulation_vars['equity_curve']) !=0 else 0.000001  #divide by zero
    trough = min(simulation_vars['equity_curve'])
    return (trough - peak) / peak

def get_position_string(simulation_vars):
    s = 'Buy' if simulation_vars['direction'] == 1 else 'Sell'
    s+=':\n'
    s+='open_price: ' + str(simulation_vars['open_price']) + '\n'
    s+='close_price: ' + str(simulation_vars['close_price']) + '\n'
    s+='p_l: ' + str(simulation_vars['p_l']) + '\n'
    s+='open_time: '+ str(simulation_vars['open_time']) + '\n'
    s+='close_time: '+ str(simulation_vars['close_time']) + '\n'
    s+='duration: '+ str(simulation_vars['duration']) + '\n'
    return s

def print_position(self):
    print(get_position_string())

def get_pip_location(simulation_vars):
    ins_df = pd.read_pickle("data/instruments.pkl")
    simulation_vars['pip_location'] = ins_df['pipLocation'].iloc[-1]
    print('pip loc: ', simulation_vars['pip_location'])
    return simulation_vars['pip_location']

def test_gaps_exceed(df, minutes): # used in preprocess_data()
    # Sort the DataFrame by the 'time' column
    df = df.sort_values('time')
    # Calculate the time differences between consecutive rows
    time_diff = df['time'].diff().dt.total_seconds() / 60  # Convert to minutes
    # Check if any time difference exceeds the specified threshold
    exceeds_threshold = any(time_diff > minutes)
    return exceeds_threshold


def shard_dataframe(dataframe):
    # Convert 'time' column to datetime if it's not already
    dataframe['time'] = pd.to_datetime(dataframe['time'])
    # Group dataframe by day and split into smaller dataframes
    shards = [group for _, group in dataframe.groupby(dataframe['time'].dt.date)]    
    return shards


def preprocess_data(simulation_vars):
    mypiplocation = get_pip_location(simulation_vars)
    get_and_pickle_instrument_info(API_KEY, ACCOUNT_ID, config.experiment_config.instrument)
    # Check if necessary files exist
    files_exist = all(
        [
            os.path.exists("data/df-0.pkl"),
            #os.path.exists("data/test_df.pkl"),
            #os.path.exists("data/small_train_df.pkl"),
            os.path.exists("data/pre_train_df.pkl")
        ]
    )
    if not files_exist:
        # Data preprocessing steps...
        num_points = 80  # Total number of points
        sine_wave = np.sin(np.linspace(0, 4 * np.pi, num_points))
        # Adjusting the sine wave to oscillate around 2.0 with an amplitude of 1.0
        peak_value = 2.1
        trough_value = 1.9
        sine_wave = sine_wave * 0.1 + 2.0
        # Replace 'URL' with the actual URL of the CSV file
        url = 'https://raw.githubusercontent.com/roni762583/AUDJPY-1M/main/DAT_ASCII_AUDJPY_M1_2020.csv'
        # Read the CSV file with ';' as the delimiter
        df = pd.read_csv(url, sep=';', header=None, names=['time', 'Open', 'High', 'Low', 'bid_c','vol'], parse_dates=True)
        df = df.drop(columns=['Open', 'High', 'Low', 'vol'])
        sin_df = df.copy(deep=True)
        # Slice the DataFrame to match the length of sine_wave
        sin_df = sin_df.iloc[:len(sine_wave)]
        # Assigning the entire DataFrame to the bid_c column
        sin_df['bid_c'] = sine_wave
        # Convert the 'time' column to pandas datetime if it's not already in datetime format
        df['time'] = pd.to_datetime(df['time'])
        sin_df['time'] = pd.to_datetime(sin_df['time'])
        # number of bars for trend line
        trendline_shift = 12
        mypiplocation = get_pip_location(simulation_vars)
        #print('mypiploc:',mypiplocation)
        den = math.pow(10, mypiplocation)
        df['shifted_bid'] = df['bid_c'].shift(trendline_shift, axis=0)
        sin_df['shifted_bid'] = sin_df['bid_c'].shift(trendline_shift, axis=0)
        # change since last bar 
        df['pips_delta'] = df['bid_c'].diff()/den # az * from /
        sin_df['pips_delta'] = sin_df['bid_c'].diff()/den # az * from /
        # trendline delta tld 
        df['tl_delta'] = (df['bid_c'] - df['shifted_bid'])/den
        sin_df['tl_delta'] = (sin_df['bid_c'] - sin_df['shifted_bid'])/den
        # add case switch for other granularities
        minutes = 1 #if (granularity == "M1") else (5 if (granularity == "M5") else 1)
        # in pips per minute 
        df['tl_slope'] = df['tl_delta']/(minutes*trendline_shift)
        sin_df['tl_slope'] = sin_df['tl_delta']/(minutes*trendline_shift)
        # get rid of extra columns 
        df = df.drop(columns=['shifted_bid', 'tl_delta'])
        sin_df = sin_df.drop(columns=['shifted_bid', 'tl_delta'])
        
        df.dropna(subset=['pips_delta', 'tl_slope'], inplace=True)
        sin_df.dropna(subset=['pips_delta', 'tl_slope'], inplace=True)
        # Reset the index of df
        df = df.reset_index(drop=True)
        sin_df = sin_df.reset_index(drop=True)
        # Find the split index (70% of the total rows)
        #split_index = int(0.7 * df.shape[0])
        #small_split = int(0.0025 * df.shape[0])
        #small_next_split = int(0.0075 * df.shape[0])
        #small_start = int(0.005 * df.shape[0])
        #small_end = int(0.0075 * df.shape[0])
        # Create neat_df as the first 70% of rows of data from df
        #small_train_df = df.iloc[small_start:small_end]
        #train_df = df.iloc[:split_index]
        # Create neat_df as the first 70% of rows of data from df
        #test_df = df.iloc[split_index:]
        pre_train_df = sin_df
        # Save data to files
        #train_df.to_pickle("data/train_df.pkl")
        #test_df.to_pickle("data/test_df.pkl")
        #small_train_df.to_pickle("data/small_train_df.pkl")
        #pre_train_df.to_pickle("data/pre_train_df.pkl")
        # Load neat_df
        #pd.read_pickle("data/small_train_df.pkl")
        daily_dfs_list = shard_dataframe(df)
        days = len(daily_dfs_list)
        print('len(daily_dfs_list): ', days)
        ct = 0
        for i, df in enumerate(daily_dfs_list):
            i+=1 #exclude day zero for pretraining on sine wave
            if i==1: 
                pre_train_df.to_pickle('data/df-0.pkl')
            pkl_me = not test_gaps_exceed(df, 10) # changed to 10 min. instead of 60, only one extra df excluded
            fn = f"data/df-{i}.pkl"
            if pkl_me:
                ct+=1
                df.to_pickle(fn)
                #print('pickled df.shape: ', df.shape)
            first_date = df.iloc[0]['time'].date()
            #print(f'{first_date} gaps detected, daily df excluded i={i}')
            print(f'pickled total of {ct} dfs')
            d = days - ct
            print(f'excluded {d} days')
            return mypiplocation
    else:
        return mypiplocation

def create_neat_config():
    # Read the NEAT configuration file and create config object
    config_file = 'config/neat_config.txt'
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    return config

def pickle_best_genome(genome, df_num=None):
    if df_num is None:
        # Handle case when only one argument is passed
        # Save the best-performing genome to a file
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        best_genome_filename = f'models/best_genome{current_time}.pkl'
        with open(best_genome_filename, "wb") as best_genome_file:
            pickle.dump(genome, best_genome_file)
        return best_genome_filename
    else:
        # handle second arg
        best_genome_filename = f'best_performing_gen_per_gen/best_genome{df_num}.pkl'
        with open(best_genome_filename, "wb") as best_genome_file:
            pickle.dump(genome, best_genome_file)
        return best_genome_filename

def eval_gemones_multi(genomes, config, simulation_vars, daily_df, queue):
    nets = []
    ge = []
    processes = []
    rewards_lst = []  # To collect async results
    # Build genome network lists and start processes
    for genome_id, genome in genomes:
        genome.fitness = 0
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        # Create a copy of simulation_vars for each simulation
        local_simulation_vars = copy.deepcopy(simulation_vars)
        data_tuple = (daily_df.copy(deep=True), net, genome_id)
        # Apply asynchronously for each genome evaluation
        result = pool.apply_async(evaluate_genome_new, (queue, data_tuple, local_simulation_vars))
        # Collect the async result
        rewards_lst.append(result)

    # Get the actual results from the async tasks
    for result in rewards_lst:
        genome_id, reward = result.get()  # Get the result of the async task
        print(f"Genome ID: {genome_id} - Reward: {reward}")

    # Find the genome with the maximum reward and update their fitness
    max_reward = float('-inf')
    best_gen_id = None
    for genome_id, total_reward in rewards_lst:
        if total_reward > max_reward:
            max_reward = total_reward
            best_gen_id = genome_id

    # Update the genomes' fitness values based on rewards
    for gid, genome in genomes:
        if gid == best_gen_id:
            _ = pickle_best_genome(genome, df_num=None)
            genome.fitness = max_reward
        else:
            genome.fitness = 0.0

    # Update the generation counter
    simulation_vars['gen_num'].value += 1


# find last checkpoint
def get_highest_checkpoint():
    # Define the checkpoint directory
    checkpoint_dir = 'checkpoints/'
    # Get all the checkpoint file names
    checkpoint_files = os.listdir(checkpoint_dir)
    # Filter and extract checkpoint numbers
    checkpoint_numbers = [
        int(re.search(r'neat-checkpoint-(\d+)', file).group(1))
        for file in checkpoint_files
        if re.match(r'neat-checkpoint-\d+', file)
    ]

    if checkpoint_numbers:
        highest_checkpoint = max(checkpoint_numbers)
        return f'checkpoints/neat-checkpoint-{highest_checkpoint}'
    else:
        return None

def restore_checkpoint(filename):
    """Resumes the simulation from a previous saved point."""
    with gzip.open(filename) as f:
        generation, config, population, species_set, rndstate = pickle.load(f)
        #random.setstate(rndstate)
        return Population(config, (population, species_set, generation))

def load_population(config):
    # Check for existing checkpoints
    checkpoint_path = get_highest_checkpoint()
    #best_genome_path
    if checkpoint_path:
        print(f"Loading population from checkpoint: {checkpoint_path}")
        population = restore_checkpoint(checkpoint_path)
        # Create a directory to store checkpoints if it doesn't exist
        checkpoint_dir = 'checkpoints/'
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Set up the checkpointer to save checkpoints in the specified directory
        checkpoint_interval = 1  # Adjust the checkpoint interval as needed
        checkpointer = neat.Checkpointer(generation_interval=checkpoint_interval,
                                        filename_prefix=os.path.join(checkpoint_dir, 'neat-checkpoint-'))
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(checkpointer)
    else:
        print("No checkpoints found. Creating new population.")
        population = Population(config)
        # Create a directory to store checkpoints if it doesn't exist
        checkpoint_dir = 'checkpoints/'
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Set up the checkpointer to save checkpoints in the specified directory
        checkpoint_interval = 1  # Adjust the checkpoint interval as needed
        checkpointer = neat.Checkpointer(generation_interval=checkpoint_interval,
                                        filename_prefix=os.path.join(checkpoint_dir, 'neat-checkpoint-'))
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(checkpointer)
    return population


def get_df_pkl_lst():
    # Define the pattern to match files with a number in the filename
    pattern = 'data/df-[0-9]*.pkl'  # Will match files like 'data/df-123.pkl'
    # Retrieve the list of files that match the pattern
    files = glob.glob(pattern)
     # Sort the files based on the number in the filename
    files.sort(key=lambda x: int(re.search(r'df-(\d+)\.pkl', x).group(1)))
    return files

def run_genome_test(genome, test_df, local_simulation_vars):
    
    # creat network from genome
    def create_network(genome):
        config = create_neat_config()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        return net

    def simulate_trading(local_simulation_vars, test_df, network):
        
        def myget_pl(local_simulation_vars):
            denominator = 10 ** float(local_simulation_vars['pip_location'])
            if(local_simulation_vars['direction']==1) :
                numerator = local_simulation_vars['current_price'] -  local_simulation_vars['open_price']
            elif(local_simulation_vars['direction']==-1):
                numerator = local_simulation_vars['open_price'] - local_simulation_vars['current_price']
            else:
                return 0.0
            local_simulation_vars['pl_pips'] = numerator / denominator
            return local_simulation_vars['pl_pips']

        def get_next_observation(local_simulation_vars):
            #simulation_vars['current_step'] = already incremented in mystep()
            if ((local_simulation_vars['current_step'] < len(data)) and (local_simulation_vars['done']==False)):
                #print('hello get_next_observation()')
                current_unrealized = myget_pl(local_simulation_vars)
                #print('current_unrealized:', current_unrealized)
                current_holding = local_simulation_vars['direction']
                #print('current_step:', local_simulation_vars['current_step'])
                observation = data.iloc[local_simulation_vars['current_step']][['pips_delta', 'tl_slope']].values 
                observation = np.append(observation, [current_unrealized, current_holding])
                #print('observation: ;-)', observation)
                #observation = deNaN(observation)
            else:
                local_simulation_vars['done'] = True
                observation = deNaN([0.0, 0.0, 0.0, 0])
                print('reached end of data, last/nan obs sent: ', observation)
            return observation
            
        def myget_action(output):
            # Define the custom softmax activation function
            #print('output:',output)
            e_x = np.exp(output - np.max(output))
            output = e_x / e_x.sum(axis=0)
            #print('output::',output)
            action = np.argmax(output)
            #print('action:::', action)
            return action

        def myposition_reset(local_simulation_vars):
                local_simulation_vars['direction'] = 0
                local_simulation_vars['volume'] = 0.0
                local_simulation_vars['equity_curve'] = [0]
                local_simulation_vars['mdd'] = 0.0
                local_simulation_vars['p_l'] = 0.0

        def myget_position_json(local_simulation_vars):
            
            # Check and handle NaT or None for open_time
            if pd.isnull(local_simulation_vars['open_time']):
                open_time_str = pd.Timestamp(0).strftime('%Y-%m-%d %H:%M:%S')  # Beginning of the epoch
            else:
                open_time_str = local_simulation_vars['open_time'].strftime('%Y-%m-%d %H:%M:%S')

            # Check and handle NaT or None for close_time
            if pd.isnull(local_simulation_vars['close_time']):
                close_time_str = pd.Timestamp(0).strftime('%Y-%m-%d %H:%M:%S')  # Beginning of the epoch
            else:
                close_time_str = local_simulation_vars['close_time'].strftime('%Y-%m-%d %H:%M:%S')

            position_data = {
                'direction': local_simulation_vars['direction'],
                'volume': local_simulation_vars['volume'],
                'open_price': local_simulation_vars['open_price'],
                'close_price': local_simulation_vars['close_price'],
                'p_l': local_simulation_vars['p_l'],
                'open_time': open_time_str, #local_simulation_vars['open_time'].strftime('%Y-%m-%d %H:%M:%S'),  # Convert to string
                'close_time': close_time_str, #local_simulation_vars['close_time'].strftime('%Y-%m-%d %H:%M:%S'),  # Convert to string
                'duration': str(local_simulation_vars['duration']),  # Convert to string
                'above_water_fraction': str(local_simulation_vars['above_water_fraction'])
            }
            jsn = json.dumps(position_data)
            #print('myget_position_json(): ', jsn)
            return jsn

        def myclose_position(local_simulation_vars):

                #print('open_time at close = ', local_simulation_vars['open_time'])

                # update pos.
                #print('vol1, dir == ', local_simulation_vars['volume'], ', ', local_simulation_vars['current_step'], flush=True)
                myupdate(local_simulation_vars)
                local_simulation_vars['close_time'] = local_simulation_vars['current_timestamp']
                local_simulation_vars['close_price'] = local_simulation_vars['current_price']
                #print('vol4, dir == ', local_simulation_vars['volume'], ', ', local_simulation_vars['current_step'], flush=True)
                jsn = myget_position_json(local_simulation_vars)
                myposition_reset(local_simulation_vars)
                return jsn

        def myopen_position(order_type_str, local_simulation_vars):
            if local_simulation_vars['volume'] == 0:
                local_simulation_vars['open_time'] = local_simulation_vars['current_timestamp']
                local_simulation_vars['open_price'] = local_simulation_vars['current_price']
                local_simulation_vars['volume'] = 100 if order_type_str=='Buy' else (-100 if order_type_str=='Sell' else 0)

        def myupdate(local_simulation_vars):
            # Initialize lwm with a default value
            lwm = None

            # set direction by volume
            local_simulation_vars['direction'] = 1 if local_simulation_vars['volume'] > 0 else (-1 if local_simulation_vars['volume'] < 0 else 0)
            
            # skip update if no position, i.e. volume is zero
            if local_simulation_vars['volume'] == 0:
                #print('volume equals zero')
                return
            
            den = local_simulation_vars['total_ticks'] if local_simulation_vars['total_ticks'] > 0 else 1
            local_simulation_vars['underwater_fraction'] = local_simulation_vars['ticks_underwater'] / den
            local_simulation_vars['p_l'] = myget_pl(local_simulation_vars)
            local_simulation_vars['duration'] = local_simulation_vars['current_timestamp'] - local_simulation_vars['open_time']
            
            # update ticks underwater
            if local_simulation_vars['p_l'] < 0:
                local_simulation_vars['ticks_underwater'] += 1
            
            # update ticks abovewater
            if local_simulation_vars['p_l'] > 0:
                local_simulation_vars['ticks_abovewater'] += 1

            # update total ticks
            local_simulation_vars['total_ticks'] += 1

            # underwater_fraction
            local_simulation_vars['underwater_fraction'] = local_simulation_vars['ticks_underwater'] / local_simulation_vars['total_ticks']

            # above_water_fraction
            den = local_simulation_vars['total_ticks'] if local_simulation_vars['total_ticks'] > 0 else 1
            local_simulation_vars['above_water_fraction'] = local_simulation_vars['ticks_abovewater'] / den

            # hwm
            if local_simulation_vars['p_l'] > local_simulation_vars['hwm']:
                local_simulation_vars['hwm'] = local_simulation_vars['p_l']

            # append to equity curve
            local_simulation_vars['equity_curve'].append(local_simulation_vars['equity_curve'][-1] + local_simulation_vars['p_l'])

            # drawdown
            # drawdown = self.hwm - self.p_l
            drawdown = calculate_drawdown(local_simulation_vars)
            if drawdown < local_simulation_vars['mdd']:  # assumes dd is negative, need to check it
                local_simulation_vars['mdd'] = drawdown

            # lwm
            if lwm is None or local_simulation_vars['p_l'] < lwm:
                lwm = local_simulation_vars['p_l']

            # update metrics
            if local_simulation_vars['underwater_fraction'] == 0:  # need to reexamine
                local_simulation_vars['underwater_fraction'] = 1
            local_simulation_vars['profit_over_underwater_fraction'] = local_simulation_vars['p_l'] / local_simulation_vars['underwater_fraction']
            local_simulation_vars['profit_over_mdd'] = local_simulation_vars['p_l'] / local_simulation_vars['mdd']
            local_simulation_vars['profit_over_underwater_fraction_and_mdd'] = local_simulation_vars['profit_over_underwater_fraction'] / local_simulation_vars['mdd']
            # need to update balance too...
        # end myupdate()

        trades_list = []
        data = test_df.copy()
        local_simulation_vars['done'] = local_simulation_vars['current_step'] >= len(data)
        print('test_df.shape', test_df.shape)
        # Main episode loop
        while not local_simulation_vars['done']:  
            #print('local_simulation_vars[current_step]:', local_simulation_vars['current_step'])        
            local_simulation_vars['current_timestamp'] = data.loc[data.index[local_simulation_vars['current_step']], 'time'] #data.loc[local_simulation_vars['current_step'], 'time'] 
            local_simulation_vars['current_price'] =  data['bid_c'].iloc[local_simulation_vars['current_step']] #data['bid_c'].iloc[local_simulation_vars['current_step']]
            
            observation = get_next_observation(local_simulation_vars)
            output = network.activate(observation)
            action = myget_action(output)

            # Execute the selected action (0=buy, 1=sell, 2=close, 3=no action)
            if action == 0:  # Buy
                # if no position open, go ahead and open simulated long position
                if local_simulation_vars['volume'] == 0:
                    myopen_position('Buy',local_simulation_vars)
                    
                else:
                    # update existing position
                    myupdate(local_simulation_vars)
            elif action == 1:  # Sell
                # if no position open, go ahead and open simulated long position
                if local_simulation_vars['volume'] == 0:
                    myopen_position('Sell',local_simulation_vars)
                else:
                    # update existing position
                    myupdate(local_simulation_vars)
            elif action == 2:  # Close
                #logging.info('action == 2 yippie yay')
                # if position open, go ahead and close it
                if local_simulation_vars['volume'] != 0:
                    # close position
                    jsn = myclose_position(local_simulation_vars)
                    # append closed position to trades list
                    #logging.info('close pos. jsn: %s', jsn)
                    trades_list.append(jsn)
                    # Parse the JSON string into a Python object
                    position_json = json.loads(jsn)
                    # Access the above_water_fraction value from the received JSON object
                    above_water_fraction = position_json['above_water_fraction']
                    pips_earned = position_json['p_l']
                    #print('closed pos. P/L: ', pips_earned)
                else:  # no position open
                    pass
            elif action == 3:  # No action
                # if position open, update it
                if local_simulation_vars['volume'] != 0:
                    myupdate(local_simulation_vars)
                else:  # no position
                    myupdate(local_simulation_vars)
            # Update the environment state (current step)
            local_simulation_vars['current_step']+=1
            # update done
            local_simulation_vars['done'] = local_simulation_vars['current_step'] >= len(data)
            
        else:  # If episode is done
            print('trades_list') 
            return trades_list
    # end simulate_trading()
        
    def plot_trades(trades, neat_df, local_simulation_vars):
        # Initialize lists to store trade information
        open_prices = []
        close_prices = []
        open_times = []
        close_times = []
        colors = []  # Store line colors
        pnl = []  # Store individual trade profit/loss

        for trade in trades:
            trade_info = json.loads(trade)
            open_prices.append(trade_info['open_price'])
            close_prices.append(trade_info['close_price'])
            open_times.append(pd.to_datetime(trade_info['open_time']))
            close_times.append(pd.to_datetime(trade_info['close_time']))

            # Determine the color based on trade profitability
            if trade_info['p_l'] > 0:
                colors.append('g')  # Green for profitable trades
            else:
                colors.append('r')  # Red for losing trades

            # Append individual trade profit/loss to the pnl list
            pnl.append(trade_info['p_l'])

        # Create a new figure for the trades
        plt.figure(figsize=(12, 6))

        # Plot the lines connecting open and close prices with different colors
        for i in range(len(open_prices)):
            plt.plot([open_times[i], close_times[i]], [open_prices[i], close_prices[i]], marker='o', color=colors[i])

            # Add individual trade profit/loss and close time as text on the chart
            #plt.text(close_times[i], close_prices[i], f'P/L: {pnl[i]:.2f}', fontsize=12, ha='center', color='g' if pnl[i] > 0 else 'r')
            #plt.text(close_times[i], close_prices[i] - 0.005, f'Close: {close_times[i].strftime("%Y-%m-%d %H:%M")}', fontsize=12, ha='center', color='blue')

        # Set labels and title
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Trades Plot')

        # Include neat_df['bid_c'] as the baseline price line in blue
        plt.plot(neat_df['time'], neat_df['bid_c'], 'b--', label='bid_c')

        # Calculate and display the total profit/loss at the top of the chart
        total_pnl = sum(pnl)
        plt.text(neat_df['time'].min(), neat_df['bid_c'].max() - 0.01, f'Total P/L: {total_pnl:.2f}', fontsize=12, ha='left', va='top', color='g' if total_pnl >= 0 else 'r')

        # Show the chart
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()

        # Get the current timestamp
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # save the plot as an image file
        gen = local_simulation_vars['gen_num']
        plot_file_name = f'graphs/{gen}.png'
        plt.savefig(plot_file_name)

    test_genome = genome #load_best_genome('best_genome.pkl')

    network = create_network(test_genome)
    
    # run data observations through network to get actions, rewards, and trades list
    trades_list = simulate_trading(local_simulation_vars, test_df, network)
    #print('returned trades list: \n', trades_list)
    # Get the current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    gen = local_simulation_vars['gen_num']
    # Create the file name with the current timestamp
    file_name = f'trades/best_gen_trades_lst_gen_{gen}.txt'
    
    # Save the trades_list to a file with the timestamped name
    with open(file_name, 'w') as file:
        for trade in trades_list:
            file.write(str(trade) + '\n')

    # Use the `plot_trades` function to plot the trades with different colors (saves image)
    if trades_list:
        plot_trades(trades_list, test_df)

# Define a wrapper function to pass additional arguments
def evaluate_genomes_wrapper(genomes, config):
    eval_gemones_multi(genomes, config, simulation_vars, daily_df, queue)


def render(self):
    # Implement rendering or visualization code if needed
    pass

# main loop
def main():
    
    # get pip loc., and shards data into df-1,..., df-n
    simulation_vars['mypiplocation'] = preprocess_data(simulation_vars)
    config = create_neat_config()
    population = load_population(config)

    # train/test nested loop
    sorted_list_of_dfs = get_df_pkl_lst()
    
    window_size = 2  # Define the size of the rolling window
    if len(sorted_list_of_dfs) >= window_size:
        # loop over daily df's
        for i in range(0, len(sorted_list_of_dfs) - window_size + 1):
            train_df_path = sorted_list_of_dfs[i]
            test_df_path = sorted_list_of_dfs[i + 1]
            train_df = pd.read_pickle(train_df_path)
            test_df = pd.read_pickle(test_df_path)

            print('crrently processing ', train_df_path)
            winner = population.run(evaluate_genomes_wrapper, n)
            print(f'winner for {train_df_path} is ... ', winner)
            best_genome_filename = pickle_best_genome(winner)
            run_genome_test(winner, test_df, simulation_vars)

            # Move the window forward
            train_df_path = test_df_path
            test_df_path = sorted_list_of_dfs[i + window_size]


# entry point
if __name__ == '__main__':
    main()
