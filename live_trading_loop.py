# demo live trading loop

# observation is incorrect, showing position when there is none:
'''
No open trades in the account.
No open positions in the account.
obs.  [-1.099999999999568 -0.12500000000000475 -0.0063 -1.0]
'''


#!pip install neat-python
import pandas as pd
import neat
import numpy as np
import json
import os
import shutil
#from google.colab import drive
import pickle
import time
import requests
import math




def fetch_candles(pair_name, count, granularity):
    url = f"{OANDA_URL}/instruments/{pair_name}/candles"
    params = dict(
        count = count,
        granularity = granularity,
        price = "MBA"
    )
    response = session.get(url, params=params, headers=SECURE_HEADER)
    return response.status_code, response.json()

def get_candles_df(json_response):
    prices = ['mid', 'bid', 'ask']
    ohlc = ['o', 'h', 'l', 'c']
    our_data = []
    for candle in json_response['candles']:
        if candle['complete'] == False:
            continue
        new_dict = {}
        new_dict['time'] = candle['time']
        new_dict['volume'] = candle['volume']
        for price in prices:
            for oh in ohlc:
                new_dict[f"{price}_{oh}"] = candle[price][oh]
        our_data.append(new_dict)
    return pd.DataFrame.from_dict(our_data)

def save_file(candles_df, pair, granularity):
    candles_df.to_pickle(f"{pair}_{granularity}.pkl")

def create_data(pair, granularity):
    code, json_data = fetch_candles(pair, candles_count, granularity)
    if code != 200:
        print(pair, "Error", flush=True)
        return
    df = get_candles_df(json_data)
    #print(f"{pair} loaded {df.shape[0]} candles from {df.time.min()} to {df.time.max()}")
    save_file(df, pair, granularity)

# utils
def get_his_data_filename(pair, granularity):
    return f"{pair}_{granularity}.pkl"

def buy_market():
    # Order payload for a market order
    order_data = {
        "order": {
            "units": "100",  # Replace with the desired amount or quantity
            "instrument": instrument,  # Replace with the desired trading instrument
            "timeInForce": "FOK",  # Fill or Kill - Ensures the entire order is filled at the best available price
            "type": "MARKET",  # Market order type
            "positionFill": "DEFAULT"  # Default position fill policy
        }
    }
    # Send the POST request to create the order
    endpoint = f"{OANDA_URL}/accounts/{account_id}/orders"
    # Send the POST request to create the order
    response = requests.post(endpoint, headers=headers, data=json.dumps(order_data))
    if response.status_code == 201:
        print("Market order placed successfully!", flush=True)
        order_info = response.json()
        print("Order ID:", order_info['orderFillTransaction']['id'], flush=True)
    else:
        print(f"Failed to place market order. Status code: {response.status_code}", flush=True)
        print(response.text, flush=True)  # Print the error message or response content for further details

def sell_market():
    # Order payload for a market order
    order_data = {
        "order": {
            "units": "-100",  # Replace with the desired amount or quantity
            "instrument": instrument,  # Replace with the desired trading instrument
            "timeInForce": "FOK",  # Fill or Kill - Ensures the entire order is filled at the best available price
            "type": "MARKET",  # Market order type
            "positionFill": "DEFAULT"  # Default position fill policy
        }
    }
    # Send the POST request to create the order
    endpoint = f"{OANDA_URL}/accounts/{account_id}/orders"
    response = requests.post(endpoint, headers=headers, data=json.dumps(order_data))
    if response.status_code == 201:
        print("Market order placed successfully!", flush=True)
        order_info = response.json()
        print("Order ID:", order_info['orderFillTransaction']['id'], flush=True)
    else:
        print(f"Failed to place market order. Status code: {response.status_code}", flush=True)
        print(response.text, flush=True)  # Print the error message or response content for further details

'''
def close_position(account_id, trade_id):
    # API endpoint for closing a position
    endpoint = f"https://api-fxtrade.oanda.com/v3/accounts/{account_id}/trades/{trade_id}/close"

    # Request headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    try:
        # Retrieve open trades to verify the status of the trade to be closed
        trades_endpoint = f"https://api-fxtrade.oanda.com/v3/accounts/{account_id}/openTrades"
        trades_response = requests.get(trades_endpoint, headers=headers)

        if trades_response.status_code == 200:
            trades_data = trades_response.json()
            trades = trades_data.get('trades', [])
            trade_to_close = None

            # Find the specific trade to be closed
            for trade in trades:
                if trade['instrument'] == instrument_to_close:
                    trade_to_close = trade
                    break

            if trade_to_close:
                units_to_close = trade_to_close.get('currentUnits')  # Get the units of the trade to close

                # Close the trade only if it's open and has units
                if units_to_close != 0:
                    close_payload = {
                        'units': str(units_to_close) if units_to_close > 0 else str(-units_to_close)
                    }

                    # Send the PUT request to close the trade
                    response = requests.put(trades_endpoint, headers=headers, json=close_payload)

                    if response.status_code == 200:
                        print(f"Trade for {instrument_to_close} closed successfully!")
                        return True
                    else:
                        print(f"Failed to close trade for {instrument_to_close}. Status code: {response.status_code}")
                        print(response.text)  # Print the error message or response content for further details
                else:
                    print(f"Trade for {instrument_to_close} is already closed.")
            else:
                print(f"Trade for {instrument_to_close} not found or already closed.")
        else:
            print(f"Failed to retrieve open trades. Status code: {trades_response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")

    return False
'''

def close_trade():
    # API endpoint for closing a trade
    endpoint = f"https://api-fxtrade.oanda.com/v3/accounts/{account_id}/trades/{trade_id}/close"
    # Request headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    try:
        # Retrieve open trades to verify the status of the trade to be closed
        trades_endpoint = f"https://api-fxtrade.oanda.com/v3/accounts/{account_id}/openTrades"
        trades_response = requests.get(trades_endpoint, headers=headers)
        if trades_response.status_code == 200:
            trades_data = trades_response.json()
            trades = trades_data.get('trades', [])
            trade_to_close = None
            # Find the specific trade to be closed
            for trade in trades:
                if trade['id'] == trade_id:
                    trade_to_close = trade
                    break
            if trade_to_close:
                units_to_close = trade_to_close.get('currentUnits')  # Get the units of the trade to close
                # Close the trade only if it's open and has units
                if units_to_close and units_to_close != '0':
                    close_payload = {
                        'units': str(units_to_close) if int(units_to_close) > 0 else str(-int(units_to_close))
                    }
                    # Send the PUT request to close the trade
                    response = requests.put(endpoint, headers=headers, json=close_payload)
                    if response.status_code == 200:
                        print(f"Trade ID: {trade_id} closed successfully!", flush=True)
                        return True
                    else:
                        print(f"Failed to close trade ID: {trade_id}. Status code: {response.status_code}", flush=True)
                        print(response.text, flush=True)  # Print the error message or response content for further details
                else:
                    print(f"Trade ID: {trade_id} is already closed.", flush=True)
            else:
                print(f"Trade ID: {trade_id} not found or already closed.", flush=True)
        else:
            print(f"Failed to retrieve open trades. Status code: {trades_response.status_code}", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
    return False

def get_action(output):
        """Get the action with the highest probability."""
        #print('hello from get_action()')
        output = softmax_activation(output)
        action = np.argmax(output)
        #print('action', action)
        return action

# Define the custom softmax activation function
def softmax_activation(x):
    """Custom softmax activation function."""
    #print('hello from softmax_activation()')
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)




print('Hello from live_trading_loop |:-{) ', flush=True)

live_trading = True
# Your OANDA API access token and account ID
access_token = "dc209942c46fbd6dd5ebe97a7043edb9-97d4122a6dfa0af915ee92ce54628997"

account_id = "001-001-5922470-001"

# API endpoints
trades_endpoint = f"https://api-fxtrade.oanda.com/v3/accounts/{account_id}/openTrades"
positions_endpoint = f"https://api-fxtrade.oanda.com/v3/accounts/{account_id}/openPositions"

headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

def get_instruments_data_filename():
    return "data/instruments.pkl"

# get last live bars as neat_df
ins_df = pd.read_pickle(get_instruments_data_filename()) #"data/instruments.pkl")

count = 15
instrument = "AUD_JPY"
granularity = "M1"
candles_count = 15
params = dict(
    count = count,
    granularity = granularity,
    price = "MBA"
)

session = requests.Session()

# prepare live observation
current_unrealized = 0.0
current_holding = 0

instrument = "AUD_JPY"

OANDA_URL = "https://api-fxtrade.oanda.com/v3"
inst_url = f"{OANDA_URL}/accounts/{account_id}/instruments"
candles_url =  f"{OANDA_URL}/instruments/{instrument}/candles"
SECURE_HEADER = {
    'Authorization': f'Bearer {access_token}'
}

instrument_to_close = instrument  # Replace with the instrument you want to close

file_path = 'gems/gem2/best_genome.pkl'

# Load the genome from the pickle file
with open(file_path, 'rb') as file:
    loaded_genome = pickle.load(file)

number_of_inputs = 4
number_of_outputs = 4
config_file_path = 'config/neat_config.txt'
# Create a configuration file (replace 'config_file_path' with your config file path)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file_path)

# Create a neural network from the loaded genome
loaded_net = None
loaded_net = neat.nn.FeedForwardNetwork.create(loaded_genome, config)
if(loaded_net):
    print('loaded net', flush=True)
# Define the interval in minutes
interval_minutes = 1  # Change this to the desired interval

last_execution = time.time()  # Initialize the time of the last execution
ins_df = pd.read_pickle("data/instruments.pkl")
mypiplocation = ins_df['pipLocation'].iloc[-1]
while live_trading:
    current_time = time.localtime()
    if current_time.tm_sec != 0:
        time.sleep(1)  # Sleep for 1 second to avoid excessive CPU usage
        continue
    current_time = time.time()
    elapsed_time = current_time - last_execution

    if elapsed_time >= interval_minutes * 60:  # Check if X minutes have passed
        last_execution = current_time  # Update the last execution time
        # Perform the task here
        print('started interval update...', flush=True)
        create_data(instrument, params['granularity'])


        filename = get_his_data_filename(instrument, granularity)
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            # Read the pickle file and process the content
            df = pd.read_pickle(filename)
            # ... rest of your code ...
        else:
            print(f"The file '{filename}' is either empty or doesn't exist.")
            continue

        non_cols = ['time', 'volume']
        mod_cols = [x for x in df.columns if x not in non_cols]
        df[mod_cols] = df[mod_cols].apply(pd.to_numeric)

        # build df for neat inputs https://youtu.be/_dWRo05gHbA?si=gTF6R9t-ByYzHpP-
        # print(df.columns)
        neat_df = df[['time', 'volume', 'bid_c']].copy()
        # number of bars for trend line
        trendline_shift = 12
        neat_df['shifted_bid'] = neat_df['bid_c'].shift(trendline_shift, axis=0)
        # change since last bar
        neat_df['pips_delta'] = neat_df['bid_c'].diff()/math.pow(10, mypiplocation)

        # trendline delta tld
        neat_df['tl_delta'] = (neat_df['bid_c'] - neat_df['shifted_bid'])/math.pow(10, mypiplocation)
        # add case switch for other granulrities
        minutes = 1 if (granularity == "M1") else (5 if (granularity == "M5") else 1)
        # in pips per minute
        neat_df['tl_slope'] = neat_df['tl_delta']/(minutes*trendline_shift)
        # drop na
        #neat_df.dropna(axis=1, inplace=True)
        # Insert the "unrealized" and "holding" columns as float type with initial values at the last position
        # Assuming you have a DataFrame named 'neat_df' with a 'time' column
        neat_df['time'] = pd.to_datetime(neat_df['time'])
        # Set the 'time' column as the index
        #neat_df.set_index('time', inplace=True)
        # Drop rows that contain NaN/None ONLY in columns: 'pips_delta', 'tl_slope'
        neat_df.dropna(subset=['pips_delta', 'tl_slope'], inplace=True)
        # reset index so it starts at zero after the drop nan's
        neat_df = neat_df.reset_index(drop=True)
        #print(neat_df.tail())
        # Fetch open trades
        trades_response = requests.get(trades_endpoint, headers=headers)
        if trades_response.status_code == 200:
            trades_data = trades_response.json()
            trades = trades_data.get('trades', [])
            if trades:
                print("Open Trades:", flush=True)
                for trade in trades:
                    trade_id = trade.get('id')
                    inst = trade.get('instrument')
                    units = int(trade.get('currentUnits'))  # Convert units to an integer
                    if inst == instrument:
                        current_holding = 1 if units>0 else (-1 if units<0 else 0)
                    print(f"Trade ID: {trade_id}, Instrument: {inst}, Units: {units}", flush=True)
            else:
                print("No open trades in the account.", flush=True)
                units = 0
        else:
            print(f"Failed to fetch open trades. Status code: {trades_response.status_code}", flush=True)

        # Fetch open positions
        positions_response = requests.get(positions_endpoint, headers=headers)
        if positions_response.status_code == 200:
            positions_data = positions_response.json()
            positions = positions_data.get('positions', [])
            if positions:
                print("\nOpen Positions:", flush=True)
                for position in positions:
                    inst = position.get('instrument')
                    unrealized_pl = position.get('unrealizedPL')
                    if inst == instrument:
                        current_unrealized = float(unrealized_pl)
                    print(f"Instrument: {inst}, Unrealized P/L: {unrealized_pl}", flush=True)
            else:
                print("No open positions in the account.", flush=True)
                unrealized_pl = 0.0
        else:
            print(f"Failed to fetch open positions. Status code: {positions_response.status_code}", flush=True)

        # get unreal. & holding from vars instead of df
        observation = neat_df.iloc[neat_df.shape[0]-1][['pips_delta', 'tl_slope']].values #, 'unrealized', 'holding']].values
        observation = np.append(observation, [current_unrealized, current_holding])
        #observation = self.deNaN(observation)
        print('obs. ', observation, flush=True)
        output = loaded_net.activate(observation)
        print('output', output, flush=True)
        action = get_action(output)
        print('action: ', action, flush=True)
        # Execute the selected action (0=buy, 1=sell, 2=close, 3=no action)
        if action == 0:  # Buy
            print('new action: OPEN BUY ', action, flush=True)
            # if no position open, go ahead and open simulated long position
            print('units=',units, flush=True)
            if units == 0:
                # Buy
                buy_market()
            else:
                print('cant buy already holding position', flush=True)

        elif action == 1:  # Sell
            # if no position open, go ahead and open simulated long position
            if units == 0:
                # Sell
                sell_market()
            else:
                print('cant sell already holding position', flush=True)

        elif action == 2:  # Close
            # if position open, go ahead and close it
            if units != 0:
                # close position
                close_trade()
            else:
                print('cant close - not holding position', flush=True)

        elif action == 3:  # No action
            print('new action: HOLD ON! ', action, flush=True)
            if units != 0:
                pass
            else: # no position
                pass

        # Wait for a short duration to avoid continuous checking
        time.sleep(1)  # Sleep for 1 second to avoid excessive CPU usage
    else:
        # If seconds are not zero, wait for a short duration before checking again
        time.sleep(1)  # Check again after a short interval
