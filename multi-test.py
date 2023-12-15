import os
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

import pickle

import matplotlib.pyplot as plt
#plt.switch_backend('TkAgg')  # Set the Tkinter backend for interactivity

#import visualize
import wget
import requests
import json
import config.acct_config
# # instrument, n, initial_balance, volume, spread
import config.experiment_config
from config.experiment_config import *

# Configure logging at the script level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

from config.acct_config import *
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
def deNaN(self, observation):
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

def render(self):
    # Implement rendering or visualization code if needed
    pass


def calculate_drawdown(global_vars):
    peak = max(global_vars['equity_curve']) if max(global_vars['equity_curve']) !=0 else 0.000001  #divide by zero
    trough = min(global_vars['equity_curve'])
    return (trough - peak) / peak

def get_position_string(global_vars):
    s = 'Buy' if global_vars['direction'] == 1 else 'Sell'
    s+=':\n'
    s+='open_price: ' + str(global_vars['open_price']) + '\n'
    s+='close_price: ' + str(global_vars['close_price']) + '\n'
    s+='p_l: ' + str(global_vars['p_l']) + '\n'
    s+='open_time: '+ str(global_vars['open_time']) + '\n'
    s+='close_time: '+ str(global_vars['close_time']) + '\n'
    s+='duration: '+ str(global_vars['duration']) + '\n'
    return s

def print_position(self):
    print(get_position_string())

# end of functions from trading environment

def get_pip_location(global_vars):
    ins_df = pd.read_pickle("data/instruments.pkl")
    global_vars['pip_location'] = ins_df['pipLocation'].iloc[-1]
    print('pip loc: ', global_vars['pip_location'])
    return global_vars['pip_location']

def preprocess_data(global_vars):

    mypiplocation = get_pip_location(global_vars)

    # Check if necessary files exist
    files_exist = all(
        [
            os.path.exists("data/train_df.pkl"),
            os.path.exists("data/test_df.pkl"),
            os.path.exists("data/small_train_df.pkl"),
            os.path.exists("data/pre_train_df.pkl")
        ]
    )

    if not files_exist:
        # Data preprocessing steps...
        # sine wave csv
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
        
        # Calculate the time differences (time deltas)
        time_diffs = df['time'].diff()

        # number of bars for trend line
        trendline_shift = 12
        mypiplocation = get_pip_location(global_vars)
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

        #print(df.shape, ' b4 drop na')
        #print(sin_df.shape, 'sin_df b4 drop na')

        # sin_df Drop rows that contain NaN/None ONLY in columns: 'pips_delta', 'tl_slope'
        df.dropna(subset=['pips_delta', 'tl_slope'], inplace=True)
        sin_df.dropna(subset=['pips_delta', 'tl_slope'], inplace=True)
        #print(df.shape, ' after drop na')
        #print(sin_df.shape, 'sin_df after drop na')

        # Reset the index of df
        df = df.reset_index(drop=True)
        sin_df = sin_df.reset_index(drop=True)

        # Find the split index (70% of the total rows)
        split_index = int(0.7 * df.shape[0])
        small_split = int(0.0025 * df.shape[0])
        # Create neat_df as the first 70% of rows of data from df
        small_train_df = df.iloc[:small_split]
        train_df = df.iloc[:split_index]
        # Create neat_df as the first 70% of rows of data from df
        test_df = df.iloc[split_index:]
        pre_train_df = sin_df
        #print('train_df.shape: ', train_df.shape)
        #print('small_train_df.shape: ', small_train_df.shape)
        #print('test_df.shape: ', test_df.shape)
        #print('pre_train_df.shape: ', pre_train_df.shape)
        
        # Save data to files
        train_df.to_pickle("data/train_df.pkl")
        test_df.to_pickle("data/test_df.pkl")
        small_train_df.to_pickle("data/small_train_df.pkl")
        pre_train_df.to_pickle("data/pre_train_df.pkl")
    # Load neat_df
    neat_df = pd.read_pickle("data/pre_train_df.pkl")
    #print('neat_df.shape: ', neat_df.shape)
    return mypiplocation, neat_df

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

def get_number_of_genomes(population):
    # Get all species in the population
    all_species = population.species.values()

    # Initialize a variable to count the total number of genomes
    total_genomes = 0

    # Iterate through each species
    for species in all_species:
        # Get the list of genomes within the species
        genomes_in_species = species.members
        
        # Get the count of genomes in the current species
        num_genomes_in_species = len(genomes_in_species)
        
        # Add the count of genomes in the current species to the total count
        total_genomes += num_genomes_in_species

    # Print the total number of genomes in the population
    #print("Number of genomes in the population:", total_genomes)
    return total_genomes

# main process of multiprocessing
def run_main_process(queue, population):
    print('run_main_process(): Running', flush=True)
    rewards_lst = []
    while True:
        try:
            data = queue.get(timeout=1)  # Wait for 1 second for new data
            if data is not None:
                rewards_lst.append(data)
            else:
                print('run_main_process() received None')
        except Exception as e:
            print(f"An exception occurred: {e, str(e)}")
            print('data=',data)
            break  # Break the loop on any exception

    print('run_main_process(): Done', flush=True)
    
    print('rewards list queue: ', rewards_lst)



def set_start_method():
    multiprocessing.set_start_method('fork')  # 'spawn'-for windows 'fork'-for *nix


def eval_gemones_multi(genomes, config):
    nets = []
    agents = []
    ge = []
    agents_processes_lst = []
    #queue = Queue()

    # Using multiprocessing Pool to limit the number of processes
    num_processes = multiprocessing.cpu_count() #// 2  # Limit to half the CPU cores
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Create a Manager object
    manager = Manager()
    queue = manager.Queue()
    global_vars = manager.dict()

    # Define your global variables
    global_vars['volume'] = 0.0
    global_vars['current_step'] = 0
    global_vars['done'] = False
    global_vars['balance'] = initial_balance
    global_vars['pip_location'] = mypiplocation
    global_vars['cost'] = -1*spread
    global_vars['current_price'] = 0.0
    global_vars['open_time'] = pd.NaT
    global_vars['open_price'] = 0.0
    global_vars['p_l'] = global_vars['cost']
    global_vars['close_time'] = pd.NaT
    global_vars['close_price'] = 0.0
    global_vars['duration'] = pd.Timedelta(0)
    global_vars['total_ticks'] = 0
    global_vars['ticks_underwater'] = 0
    global_vars['ticks_abovewater'] = 0
    global_vars['underwater_fraction'] = 0.0
    global_vars['above_water_fraction'] = 0.0
    global_vars['hwm'] = global_vars['cost']
    global_vars['lwm'] = global_vars['cost']
    global_vars['mdd'] = global_vars['cost']
    global_vars['profit_over_underwater_fraction'] = 0.0
    global_vars['profit_over_mdd'] = 0
    global_vars['profit_over_underwater_fraction_and_mdd'] = 0
    global_vars['equity_curve'] = [global_vars['cost']] #  list
    global_vars['direction'] = 0 # -1 short, +1 long


    # build genome and network lists  
    for genome_id, genome in genomes:
        genome.fitness = 0
        ge.append(genome)
        #trading_env = TradingEnvironment(neat_df, initial_balance, mypiplocation, None)
        #agent = NEATAgent(trading_env, genome, config)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        #print('genome_id',genome_id)
        #trading_env.net = net
        #agent.set_network(net)
        #agents.append(agent)
        
        data_tuple = (neat_df.copy(deep=True), net)
        
    # Iterate through genomes to create processes
    for _, genome in genomes:
        data_tuple = (neat_df.copy(deep=True), net)  # Update this line based on your logic
        p = Process(target=evaluate_genome, args=(queue, global_vars, data_tuple))
        p.start()
        agents_processes_lst.append(p)
    

    # Check if the lengths of both lists are equal
    if len(genomes) == len(results):
        # Create a new list to store genomes with fitness
        genomes_with_fitness = []

        # Iterate through each pair of genome and result using zip
        for (genome_id, genome_data), result in zip(genomes, results):

            # ASSIGN GENOME FITNESS
            genome_data.fitness = result
            
    else:
        print("Genomes and results lists have different lengths.")
    print('results: ', results)
    '''
    for genome, total_reward in zip(ge, total_rewards_list):
        genome.fitness = total_reward
    '''
    main_process = Process(target=run_main_process, args=(queue, population))
    main_process.start()
    queue.put(None)
    main_process.join()


# Code for evaluating a single genome
def evaluate_genome(queue, global_vars, data_tuple): # input_tuple: (neat_df, network)
    neat_df_cp, network = data_tuple
    
    data = neat_df_cp.copy()
    data_copy = neat_df_cp.copy()

    trades_list = []
    
    # Create a logger for this class
    logger = logging.getLogger(__name__)  # Using the module's name as the logger's name
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Initialize the integer
    list_for_increment_step = [0]  # Put the integer inside a list
    
    def myget_pl(global_vars):
        denominator = 10 ** float(global_vars['pip_location'])
        pl_pips = (global_vars['current_price'] - global_vars['open_price']) / denominator
        return pl_pips

    def get_next_observation(global_vars):
        global_vars['current_step'] = list_for_increment_step[0]
        if ((global_vars['current_step'] < len(data)) and (global_vars['done']==False)):
            #print('hello get_next_observation()')
            current_unrealized = myget_pl(global_vars['pip_location'])
            #print('current_unrealized:', current_unrealized)
            current_holding = global_vars['direction']
            #print('current_step:', global_vars['current_step'])
            observation = data.iloc[global_vars['current_step']][['pips_delta', 'tl_slope']].values 
            observation = np.append(observation, [current_unrealized, current_holding])
            #print('observation: ;-)', observation)
            #observation = deNaN(observation)
        else:
            global_vars['done'] = True
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

    def mystep(global_vars):
        #logging.info('hello step(), input action = %s', action)
        from config.experiment_config import volume

        def myposition_reset(global_vars):
                global_vars['direction'] = 0
                global_vars['volume'] = 0.0
                global_vars['equity_curve'] = [0]
                global_vars['mdd'] = 0.0
                global_vars['p_l'] = 0.0

        def myget_position_json(global_vars):
            position_data = {
                'direction': global_vars['direction'],
                'volume': global_vars['volume'],
                'open_price': global_vars['open_price'],
                'close_price': global_vars['close_price'],
                'p_l': global_vars['p_l'],
                'open_time': global_vars['open_time'].strftime('%Y-%m-%d %H:%M:%S'),  # Convert to string
                'close_time': global_vars['close_time'].strftime('%Y-%m-%d %H:%M:%S'),  # Convert to string
                'duration': str(global_vars['duration']),  # Convert to string
                'above_water_fraction': str(global_vars['above_water_fraction'])
            }
            jsn = json.dumps(position_data)
            print('myget_position_json(): ', jsn)
            return jsn

        def myclose_position(global_vars):
                close_time = global_vars['current_timestamp']
                close_price = global_vars['current_price']
                #print('open_time at close = ', global_vars['open_time'])

                # update pos.
                print('vol1, dir == ', global_vars['volume'], ', ', global_vars['current_step'], flush=True)
                myupdate(global_vars)
                print('vol4, dir == ', global_vars['volume'], ', ', global_vars['current_step'], flush=True)
                jsn = myget_position_json(global_vars)
                myposition_reset()
                return jsn

        def myopen_position(global_vars):
            if global_vars['volume'] == 0:
                    # position object
                    #global_vars['current_price'] = current_price
                    print('myopen_position() current_timestamp: ', current_timestamp)
                    global_vars['open_time'] = current_timestamp
                    global_vars['open_price'] = global_vars['current_price']
                    global_vars['volume'] = config.experiment_config.volume

        def myupdate(global_vars):
            #set direction by volume
            print('vol2, dir == ', global_vars['volume'], ', ', global_vars['direction'], flush=True)
            global_vars['direction'] = 1 if global_vars['volume']>0 else (-1 if global_vars['volume']<0 else 0)
            print('vol3, dir == ', global_vars['volume'], ', ', global_vars['direction'], flush=True)
            # skip update if no position, i.e. volume is zero
            if(global_vars['volume']==0):
                print('volume equals zero')
                #return
            
            #print('myupdate(); volume, direction:',global_vars['volume'],', ',direction)
            #current_price = current_price global_vars['current_price'] ?
            den = global_vars['total_ticks'] if global_vars['total_ticks']>0 else 1
            global_vars['underwater_fraction'] = global_vars['ticks_underwater'] / den
            global_vars['p_l'] = myget_pl(global_vars['pip_location'])
            duration = global_vars['current_timestamp'] - global_vars['open_time']
            # update ticks underwater
            if global_vars['p_l'] < 0:
                global_vars['ticks_underwater'] += 1
            # update ticks abovewater
            if global_vars['p_l'] > 0:
                global_vars['ticks_abovewater'] += 1

            # update total ticks
            global_vars['total_ticks'] += 1

            # underwater_fraction
            global_vars['underwater_fraction'] = global_vars['ticks_underwater'] / global_vars['total_ticks']

            # above_water_fraction
            den = global_vars['total_ticks'] if global_vars['total_ticks']>0 else 1
            global_vars['above_water_fraction'] = global_vars['ticks_abovewater'] / den

            # hwm
            if global_vars['p_l'] > global_vars['hwm']:
                global_vars['hwm'] = global_vars['p_l']

            # append to equity curve
            global_vars['equity_curve'].append(global_vars['equity_curve'][-1] + global_vars['p_l'])

            # drawdown
            #drawdown = self.hwm - self.p_l
            drawdown = calculate_drawdown(global_vars['equity_curve'])
            if drawdown < global_vars['mdd']: # assumes dd is negative, need to check it
                global_vars['mdd'] = drawdown

            # lwm
            if lwm is None or global_vars['p_l'] < lwm:
                lwm = global_vars['p_l']

            # update metrics
            if(global_vars['underwater_fraction'] ==0): #need to reexamine
                global_vars['underwater_fraction'] = 1
            global_vars['profit_over_underwater_fraction'] = global_vars['p_l'] / global_vars['underwater_fraction']
            global_vars['profit_over_mdd'] = global_vars['p_l'] / global_vars['mdd']
            global_vars['profit_over_underwater_fraction_and_mdd'] = global_vars['profit_over_underwater_fraction'] / global_vars['mdd']
        # end myupdate()

        def reward_function8(pips_earned, above_water_fraction):
            # Use root logger to check if the configuration is affecting the method
            #logging.info('reward_function8(), executing'+str(type(pips_earned)))
            reward = 0.001
            risk_adj_reward = pips_earned * float(global_vars['above_water_fraction'])
            reward += pips_earned # risk_adj_reward
            #print('reward_function8(), reward = ' + str(reward))
            return reward

        # update done
        global_vars['done'] = global_vars['current_step'] >= len(data)
        
        # if episode NOT done
        if not global_vars['done']:          
            current_timestamp = data.loc[global_vars['current_step'], 'time']
            global_vars['current_step'] = list_for_increment_step[0]
            #logging.info('current_step %s',global_vars['current_step'])
            global_vars['current_price'] = data['bid_c'].iloc[global_vars['current_step']]
            #logging.info('current_price %s',global_vars['current_price'])
            reward = 0.0  
            observation = get_next_observation(global_vars['done'])
            #print('observation+++:',observation)
            output = network.activate(observation)
            #print('output:::::',output)
            action = myget_action(output)

            # Execute the selected action (0=buy, 1=sell, 2=close, 3=no action)
            if action == 0:  # Buy
                # if no position open, go ahead and open simulated long position
                if global_vars['volume'] == 0:
                    myopen_position(global_vars)
                    
                else:
                    # update existing position
                    myupdate(global_vars)
            elif action == 1:  # Sell
                # if no position open, go ahead and open simulated long position
                if global_vars['volume'] == 0:
                    global_vars['open_price'] = global_vars['current_price']
                    global_vars['open_time'] = current_timestamp
                    global_vars['volume'] = -1 * config.experiment_config.volume
                else:
                    # update existing position
                    myupdate(global_vars)
            elif action == 2:  # Close
                #logging.info('action == 2 yippie yay')
                # if position open, go ahead and close it
                if global_vars['volume'] != 0:
                    # close position
                    jsn = myclose_position(global_vars)
                    # append closed position to trades list
                    #logging.info('jsn: %s', jsn)
                    trades_list.append(jsn)
                    # Parse the JSON string into a Python object
                    position_json = json.loads(jsn)
                    # Access the above_water_fraction value from the received JSON object
                    above_water_fraction = position_json['above_water_fraction']
                    pips_earned = position_json['p_l']
                    reward = reward_function8(pips_earned, above_water_fraction)
                    #logging.info('on close reward = %s', reward)
                else:  # no position open
                    pass
                    logging.info('cannot close, no position open')
            elif action == 3:  # No action
                # if position open, update it
                if global_vars['volume'] != 0:
                    myupdate(global_vars)
                else:  # no position
                    pass
            # Update the environment state (current step)
            global_vars['current_step']+=1
            
            global_vars['current_step'] = list_for_increment_step[0]
            global_vars['done'] = global_vars['current_step'] >= len(data)

            # Additional info (optional)
            info = {
                'balance': global_vars['balance'],
                'position_pnl': global_vars['p_l'],
                'open_price': global_vars['open_price'],
            }
            
            #logging.info('mystep(): %s\n%s\n%s\n%s', observation, reward, done, info)
            return observation, reward, global_vars['done'], info
        else:  # If episode is done
            print('trades list:', trades_list)
            return (0.0,0.0,0.0,0), 0.0, global_vars['done'], {}  # Return default values or None when the episode is done
    # end mystep()


    # loop over data and simulate trading
    total_reward = 0.0
    next_observation = get_next_observation(global_vars['done']) 
    #print('first.next_observation: ', next_observation)
    prev_net_idx = -9
    while not global_vars['done']:
        #print('next_observation=',next_observation)
        output = network.activate(next_observation)

        action = myget_action(output)
        #print('action: ', action)
        next_observation, reward, global_vars['done'], info = mystep(global_vars)
        total_reward += reward
        
    # add to the queue
    queue.put((total_reward, trades_list))#total_reward)

    return total_reward
# end evaluate_genome()

def set_network(self, input_net):
    self.net = input_net
    self.trading_env.net = self.net
    

# entry point
if __name__ == '__main__':
    trade_lists = []
    mypiplocation = np.nan
    neat_df = pd.DataFrame()
    
    get_and_pickle_instrument_info(API_KEY, ACCOUNT_ID, config.experiment_config.instrument)
    mypiplocation, neat_df = preprocess_data(global_vars)
    
    total_rewards_list = []
    
    # set star method for multiprocessing
    set_start_method()

    # get population
    config = create_neat_config()
    population = load_population(config)
    #print('population:: ', population)

    #n = 2  # Number of generations to run
    
    winner = population.run(eval_gemones_multi, n)
    '''
    #print('winner is ... ', winner)
    best_genome_filename = pickle_best_genome(winner)

    '''
