
import pandas as pd
import requests
import json
from src.neat_agent import *
from src.trading_environment import *
import neat
import os.path

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


def get_tradeslist_pnl(trades_list):
    pnl_sum = 0.0
    for trade in trades_list:
        trade_dict = json.loads(trade)
        pnl_sum += trade_dict.get('p_l', 0.0)
    return pnl_sum


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
