import os
import re
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

def preprocess_data(simulation_vars):

    mypiplocation = get_pip_location(simulation_vars)
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
        # sin_df Drop rows that contain NaN/None ONLY in columns: 'pips_delta', 'tl_slope'
        df.dropna(subset=['pips_delta', 'tl_slope'], inplace=True)
        sin_df.dropna(subset=['pips_delta', 'tl_slope'], inplace=True)
        # Reset the index of df
        df = df.reset_index(drop=True)
        sin_df = sin_df.reset_index(drop=True)
        # Find the split index (70% of the total rows)
        split_index = int(0.7 * df.shape[0])
        small_split = int(0.0025 * df.shape[0])
        small_next_split = int(0.0075 * df.shape[0])
        # Create neat_df as the first 70% of rows of data from df
        small_train_df = df.iloc[small_split:small_next_split]
        train_df = df.iloc[:split_index]
        # Create neat_df as the first 70% of rows of data from df
        test_df = df.iloc[split_index:]
        pre_train_df = sin_df
        # Save data to files
        train_df.to_pickle("data/train_df.pkl")
        test_df.to_pickle("data/test_df.pkl")
        small_train_df.to_pickle("data/small_train_df.pkl")
        pre_train_df.to_pickle("data/pre_train_df.pkl")
    # Load neat_df
    neat_df = pd.read_pickle("data/small_train_df.pkl")
    #print('neat_df.shape: ', neat_df.shape)
    return mypiplocation, neat_df

def create_neat_config():
    # Read the NEAT configuration file and create config object
    config_file = 'config/neat_config.txt'
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    return config

def pickle_best_genome(winner):
    # Save the best-performing genome to a file
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    best_genome_filename = f'models/best_genome{current_time}.pkl'
    with open(best_genome_filename, "wb") as best_genome_file:
        pickle.dump(winner, best_genome_file)
    return best_genome_filename

def set_start_method():
    # 'forkserver' # לא עובד
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn')
        print('windows detected - using spawn multiprocessing method')
    else:
        multiprocessing.set_start_method('fork')
        print('non-windows detected - using fork multiprocessing method')

def eval_gemones_multi(genomes, config):
    nets = []
    ge = []
    processes = []
    rewards_lst = []  # To collect rewards
    # Build genome network lists and start processes
    for genome_id, genome in genomes:
        genome.fitness = 0
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        # Create a copy of simulation_vars for each simulation
        local_simulation_vars = copy.deepcopy(simulation_vars)
        data_tuple = (neat_df.copy(deep=True), net, genome_id)  # Update this line based on your logic
        p = Process(target=evaluate_genome, args=(queue, data_tuple, local_simulation_vars))
        processes.append(p)
        p.start()
    # Join all simulation processes
    for p in processes:
        p.join()
    # Collect rewards
    for p in processes:
        try:
            msg = queue.get(timeout=1)
            if msg is not Empty():
                rewards_lst.extend(msg)
            else:
                break
        except Exception as e:
            #print(f"An exception occurred: {e, str(e)}")
            break  # Break the loop on any exception

    # Processing rewards_lst or any other necessary actions based on collected data
    if not rewards_lst:
        pass #print('No results received !!!\n\n\n')
    else:
        #print('Received rewards:\n', rewards_lst)
        # Process the rewards list
        
        # Iterate through each entry in rewards_lst and assign fitness to genomes
        # Reconstruct the list of tuples
        genomes_with_rewards = [(rewards_lst[i], rewards_lst[i + 1]) for i in range(0, len(rewards_lst), 2)]

        # Print the reconstructed list of tuples
        #print('Print the reconstructed list:\n')
        #print(genomes_with_rewards)
        #print('\n entries: \n')
        for entry in genomes_with_rewards:
            #print('entry: ', entry)
            genome_id = entry[0]
            total_reward = entry[1]
            # Assuming genomes is a list of tuples [(genome_id, genome), ...]
            for gid, genome in genomes:
                if gid == genome_id:
                    # Assign fitness based on total_reward to the matching genome
                    genome.fitness = total_reward
                else:
                    genome.fitness = 0.0
            
# end eval_gemones_multi()

# Code for evaluating a single genome
def evaluate_genome(queue, data_tuple, local_simulation_vars): # input_tuple: (neat_df, network)
    neat_df_cp, network, genome_id = data_tuple
    global top_genomes
    data = neat_df_cp.copy()
    #data_copy = neat_df_cp.copy()

    trades_list = []
    
    # Create a logger for this class
    logger = logging.getLogger(__name__)  # Using the module's name as the logger's name
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
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

    def mystep(local_simulation_vars):
        #logging.info('hello step(), input action = %s', action)

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
                #print('myopen_position() open_time: ', local_simulation_vars['open_time'])

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
        # end myupdate()

        def reward_function(pips_earned, above_water_fraction):
            # Use root logger to check if the configuration is affecting the method
            #logging.info('reward_function8(), executing'+str(type(pips_earned)))
            #reward = 0.001
            risk_adj_reward = pips_earned * float(local_simulation_vars['above_water_fraction'])
            reward = pips_earned #risk_adj_reward
            return reward

        # update done
        local_simulation_vars['done'] = local_simulation_vars['current_step'] >= len(data)
        
        # if episode NOT done
        if not local_simulation_vars['done']:          
            local_simulation_vars['current_timestamp'] = data.loc[local_simulation_vars['current_step'], 'time']
            #local_simulation_vars['current_step']
            #logging.info('current_step %s',local_simulation_vars['current_step'])
            local_simulation_vars['current_price'] = data['bid_c'].iloc[local_simulation_vars['current_step']]
            #logging.info('current_price %s',local_simulation_vars['current_price'])
            reward = 0.0001  
            observation = get_next_observation(local_simulation_vars)
            #print('observation+++:',observation)
            output = network.activate(observation)
            #print('output:::::',output)
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
                    reward = reward_function(pips_earned, above_water_fraction)
                    #logging.info('on close reward = %s', reward)
                else:  # no position open
                    pass
                    #logging.info('cannot close, no position open')
            elif action == 3:  # No action
                # if position open, update it
                if local_simulation_vars['volume'] != 0:
                    myupdate(local_simulation_vars)
                else:  # no position
                    myupdate(local_simulation_vars)
            # Update the environment state (current step)
            #print('b4 incr. local_simulation_vars[current_step]', local_simulation_vars['current_step'])
            local_simulation_vars['current_step']+=1
            #print('aftr. incr. local_simulation_vars[current_step]', local_simulation_vars['current_step'])
            local_simulation_vars['done'] = local_simulation_vars['current_step'] >= len(data)
            #print('len(data)=', len(data), ',  local_simulation_vars[-done-] =', local_simulation_vars['done'] )
            # Additional info (optional)
            info = {
                'balance': local_simulation_vars['balance'],
                'position_pnl': local_simulation_vars['p_l'],
                'open_price': local_simulation_vars['open_price'],
            }
            
            #logging.info('mystep(): %s\n%s\n%s\n%s', observation, reward, done, info)
            return observation, reward, local_simulation_vars['done'], info
        else:  # If episode is done
            #logging.info('trades list: ', trades_list)
            return (0.0,0.0,0.0,0), 0.0, local_simulation_vars['done'], {}  # Return default values or None when the episode is done
    # end mystep()

    # loop over data and simulate trading
    total_reward = 0.0
    next_observation = get_next_observation(local_simulation_vars) 
    # Initialize a list to keep track of the top genomes and their profits
    
    while not local_simulation_vars['done']:
        #print('next_observation=',next_observation)
        output = network.activate(next_observation)

        action = myget_action(output)
        #print('action: ', action)
        next_observation, reward, local_simulation_vars['done'], info = mystep(local_simulation_vars)
        total_reward += reward
        
    # end of simulation, zero out non-traders, and add to the queue
    # Keep track of the top genomes and their rewards
    genome_info = {'total_reward': total_reward, 'genome_id': genome_id, 'trades_list': trades_list}  # Replace current_genome_id with the actual ID of the genome
    top_genomes.append(genome_info)
    top_genomes = sorted(top_genomes, key=lambda x: x['total_reward'], reverse=True)[:10]  # Keep only the top ten genomes
    #logging.info('top_genomes: ', top_genomes)
    # write top genomes to file?
    if len(trades_list)>0:
        queue.put((genome_id, total_reward)) #, trades_list))
    else:
        #print('empty trades list - zeroing reward ')
        total_reward = 0.0
    
    return total_reward
# end evaluate_genome()


# entry point
if __name__ == '__main__':
    
    trade_lists = []
    neat_df = pd.DataFrame()
        
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
    # Define your global variables
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
        'top_genomes': top_genomes
    }
    # Add top_genomes to simulation_vars
    simulation_vars['top_genomes'] = top_genomes
    
    get_and_pickle_instrument_info(API_KEY, ACCOUNT_ID, config.experiment_config.instrument)

    simulation_vars['mypiplocation'], neat_df = preprocess_data(simulation_vars)

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
    '''
    def load_population(config):
        # Check for existing checkpoints
        checkpoint_path = get_highest_checkpoint()
        if checkpoint_path:
            print(f"Loading population from checkpoint: {checkpoint_path}")
            population =  restore_checkpoint(checkpoint_path)
        else:
            print("No checkpoints found. Creating new population.")
            population = Population(config)
        return population
        
    def load_population(config):
        # Create the NEAT population using the provided configuration
        p = neat.Population(config)
        # Create a directory to store checkpoints if it doesn't exist
        checkpoint_dir = 'checkpoints/'
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
    '''
    def load_population(config):
        # Check for existing checkpoints
        checkpoint_path = get_highest_checkpoint()
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


    def plot_neat_df(*col_names):
        import matplotlib.pyplot as plt
        import datetime
        from datetime import datetime
        # Create a new figure for the trades
        plt.figure(figsize=(12, 6))

        for col_name in col_names:
            # Plot each specified column
            plt.plot(neat_df.index, neat_df[col_name], label=col_name)

        # Set labels and title
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('NEAT DataFrame Plot')

        # Show the chart with specified columns
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()

        # Get the current timestamp
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Save the plot as an image file
        plot_file_name = f'graphs/{current_time}_{"_".join(col_names)}.png'
        plt.savefig(plot_file_name)
        plt.show()


    config = create_neat_config()
    #plot_neat_df('pips_delta', 'tl_slope')
    population = load_population(config)

    '''
    import subprocess
    def print_pwd_and_ls():
        # Execute 'pwd' command
        pwd_output = subprocess.run(['pwd'], capture_output=True, text=True)
        print("Current Directory (pwd):")
        print(pwd_output.stdout)

        # Execute 'ls -la' command
        ls_output = subprocess.run(['ls', '-la'], capture_output=True, text=True)
        print("\nDirectory Listing (ls -la):")
        print(ls_output.stdout)
    # Call the function to print pwd and ls -la
    print_pwd_and_ls()
    '''
    winner = population.run(eval_gemones_multi, n)
    
    print('winner is ... ', winner)
    best_genome_filename = pickle_best_genome(winner)