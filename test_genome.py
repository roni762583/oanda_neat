# this code will run winner genome on test data, and graph the trades
import numpy as np
import pickle
import neat
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from config.experiment_config import *
import json
import datetime
from datetime import datetime, timedelta

def load_best_genome(filename):
    best_genome_path = f'models/{filename}'
    try:
        with open(best_genome_path, 'rb') as file:
            best_genome = pickle.load(file)
        return best_genome
    except FileNotFoundError:
        print(f"File {best_genome_path} not found.")
        return None

def create_neat_config():
    # Read the NEAT configuration file and create config object
    config_file = 'config/neat_config.txt'
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    return config

# creat network from genome
def create_network(genome):
    config = create_neat_config()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net

# define test data

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
        # Create neat_df as the first 70% of rows of data from df
        small_train_df = df.iloc[:small_split]
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



def simulate_trading(local_simulation_vars, neat_df_cp, network):
    
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

    def reward_function8(pips_earned, above_water_fraction):
        # Use root logger to check if the configuration is affecting the method
        #logging.info('reward_function8(), executing'+str(type(pips_earned)))
        #reward = 0.001
        risk_adj_reward = pips_earned * float(local_simulation_vars['above_water_fraction'])
        reward = pips_earned #risk_adj_reward
        #print('reward_function8(), reward = ' + str(reward))
        return reward


    trades_list = []
    data = neat_df_cp.copy()
    local_simulation_vars['done'] = local_simulation_vars['current_step'] >= len(data)
    print('neat_df.shape', neat_df.shape)
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
    

def plot_trades(trades, neat_df):
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
        plt.text(close_times[i], close_prices[i], f'P/L: {pnl[i]:.2f}', fontsize=12, ha='center', color='g' if pnl[i] > 0 else 'r')
        plt.text(close_times[i], close_prices[i] - 0.005, f'Close: {close_times[i].strftime("%Y-%m-%d %H:%M")}', fontsize=12, ha='center', color='blue')

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
    plot_file_name = f'graphs/{current_time}.png'
    plt.savefig(plot_file_name)
    plt.show()



# entry point
if __name__ == '__main__':
    
    trade_lists = []
    neat_df = pd.DataFrame()    
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
        'direction': 0  # -1 short, +1 long
    }
    
    get_and_pickle_instrument_info(API_KEY, ACCOUNT_ID, instrument)

    simulation_vars['mypiplocation'], neat_df = preprocess_data(simulation_vars)
    
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

    def print_models_directory():
        # Execute 'ls -la' command in the models directory
        ls_output = subprocess.run(['ls', '-la', '/app/models'], capture_output=True, text=True)
        print("\nContents of /app/models:")
        print(ls_output.stdout)
    # Call the function to print pwd and ls -la
    print_pwd_and_ls()
    # Call the function to print the contents of /app/models directory
    print_models_directory()
    '''
    # load models/best_genome.pkl
    test_genome = load_best_genome('best_genome.pkl')

    network = create_network(test_genome)
    
    # run data observations through network to get actions, rewards, and trades list
    trades_list = simulate_trading(simulation_vars, neat_df, network)
    #print('returned trades list: \n', trades_list)
    # Get the current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create the file name with the current timestamp
    file_name = f'trades/trades_list_{current_time}.txt'
    
    # Save the trades_list to a file with the timestamped name
    with open(file_name, 'w') as file:
        for trade in trades_list:
            file.write(str(trade) + '\n')

    # Use the `plot_trades` function to plot the trades with different colors (saves image)
    if trades_list:
        plot_trades(trades_list, neat_df)