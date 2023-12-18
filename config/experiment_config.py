
from config.acct_config import API_KEY, ACCOUNT_ID

instrument = "AUD_JPY"

OANDA_URL = "https://api-fxtrade.oanda.com/v3"

inst_url = f"{OANDA_URL}/accounts/{ACCOUNT_ID}/instruments"

candles_url =  f"{OANDA_URL}/instruments/{instrument}/candles"

SECURE_HEADER = {
    'Authorization': f'Bearer {API_KEY}'
}

spread = 1.5 # pips (for cost calc.)

volume = 100

initial_balance = 1000

# Run for up to n generations. eval_genomes acts as fitness function
n = 2

# Define reward parameters
gamma = 0.95                   # reward_fuction()
exploration_bonus = 0.001      # reward_fuction()
maximal_drawdown_penalty = 0.1 # reward_fuction()
