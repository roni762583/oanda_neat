
import numpy as np
#from src.position import *
from config.experiment_config import volume
import logging
import random
from config.experiment_config import spread
import json
import pandas as pd

# Configure logging at the script level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TradingEnvironment:

    # INIT CONSTRUCTOR FOR TRADING ENVIRONMENT
    def __init__(self, data, initial_balance, pip_location, neural_network):
        self.current_step = 0
        self.data = data.copy()
        self.data_copy = data.copy()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.pip_location = pip_location
        #self.position = self.Position(0,0,pip_location)
        self.trades_list = []
        #print('about to initialize done')
        self.done = False  # Initialize the 'done' attribute
        self.net = neural_network  # Accept the neural network as an argument
        self.ticks = 0 # to hold total number of price changes a position underwent
        self.training_level = 1 # =3 is with risk adj. returns, any other value runs levels 1 & 2 combined (order of ops., pnl)
        # Create a logger for this class
        self.logger = logging.getLogger(__name__)  # Using the module's name as the logger's name
        
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # PARAMETERS FROM PREV. POSITION CLASS
        cost = -1*spread
        # negative volume represents a short trade
        self.volume = volume
        self.current_price = 0.0
        self.open_time = pd.Timestamp.now()
        self.open_price = 0.0
        self.pip_location = pip_location
        self.p_l = cost
        self.close_time = pd.Timestamp.now()
        self.close_price = None
        self.duration = 0
        self.total_ticks = 0
        self.ticks_underwater = 0
        self.ticks_abovewater = 0
        self.underwater_fraction = 0.0
        self.above_water_fraction = 0.0
        self.hwm = cost
        self.lwm = cost
        self.mdd = cost # Initialize MDD to the initial spread
        self.pip_location = pip_location
        # metrics
        self.profit_over_underwater_fraction = 0
        self.profit_over_mdd = 0
        self.profit_over_underwater_fraction_and_mdd = 0
        # updated mdd calc
        self.equity_curve = [cost]  # Initialize equity curve with the initial spread
        self.direction = 0 # -1 short, +1 long

    def reset(self):
        # Reset environment state
        self.data = self.data_copy
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_reset()
        self.open_price = 0.0
        self.open_time = None
        self.time_underwater = 0
        self.pl_high_water_mark = 0.0
        self.pl_low_water_mark = 0.0
        #print('hello trd env reset()')
        self.done = False  # reset the 'done' attribute
        self.ticks = 0     
        # Return initial observation
        return self._next_observation() #, self.current_step

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

    # Define the custom softmax activation function
    def softmax_activation(self, x):
        """Custom softmax activation function."""
        #print('hello from softmax_activation()')
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_action(self, output):
        """Get the action with the highest probability."""
        output = self.softmax_activation(output)
        action = np.argmax(output)
        #if action==2:
        #    logging.info('action: %s',action)
        return action

    def step(self, action):
        
        # Use the logger object to log messages
        #self.logger.info('hello step(), input action = %s', action)

        # update self.done
        self.done = self.current_step >= len(self.data)

        if self.done:  # If episode is done
            return None, 0, self.done, {}  # Return default values or None when the episode is done
        else:          # if episode NOT done
            current_time = self.data.loc[self.current_step, 'time']
            current_price = self.data['bid_c'].iloc[self.current_step]
            reward = 1.0  # Initialize reward for this step
            # Pass observation through the neural network to get trading decision
            observation = self._next_observation()
            # get agent / network output
            output = self.net.activate(observation)
            # get action from output
            action = self.get_action(output)
            #print('action=',action)
            #logging.info('observation: %s, output: %s, action: %s', observation, output, action)
            # Execute the selected action (0=buy, 1=sell, 2=close, 3=no action)
            if action == 0:  # Buy
                # if no position open, go ahead and open simulated long position
                if self.volume == 0:
                    # position object
                    self.current_price = current_price
                    self.open_time = current_time
                    self.open_price = current_price
                    self.volume = volume
                else:
                    # update existing position
                    self.update(self.current_step, current_time, current_price)
            elif action == 1:  # Sell
                # if no position open, go ahead and open simulated long position
                if self.volume == 0:
                    self.current_price = current_price
                    self.open_price = current_price
                    self.open_time = current_time
                    self.volume = -1 * volume
                else:
                    # update existing position
                    self.update(self.current_step, current_time, current_price)
            elif action == 2:  # Close
                # if position open, go ahead and close it
                if self.volume != 0:
                    # close position
                    jsn = self.close_position(current_price, current_time, self.current_step)
                    # append closed position to trades list
                    self.trades_list.append(jsn)
                    # Parse the JSON string into a Python object
                    position_json = json.loads(jsn)
                    # Access the above_water_fraction value from the received JSON object
                    above_water_fraction = position_json['above_water_fraction']
                    pips_earned = position_json['p_l']
                    reward = self.reward_function5(pips_earned, above_water_fraction)
                    logging.info('close reward = ', reward) #%s', reward)
                else:  # no position open
                    pass
                    #logging.info('cannot close, no position open')
            elif action == 3:  # No action
                # if position open, update it
                if self.volume != 0:
                    self.update(self.current_step, current_time, current_price)
                else:  # no position
                    pass

            # Update the environment state (current step)
            self.current_step += 1
            self.done = self.current_step >= len(self.data)

            # Additional info (optional)
            info = {
                'balance': self.balance,
                'position_pnl': self.p_l,
                'open_price': self.open_price,
            }
            
            #reward = 12.7 # this line doesn't affect fitness!!!
            #logging.info('step(): %s\n%s\n%s\n%s', observation, reward, self.done, info)
            return observation, reward, self.done, info
        

    def render(self):
        # Implement rendering or visualization code if needed
        pass


    def _next_observation(self):
        #print('hello from _next...')
        #print('self.current_step',self.current_step)
        #print('len(self.data)',len(self.data))
        #print('self.done',self.done)
        if ((self.current_step < len(self.data)) and (self.done==False)):
            #print('hello again')
            current_unrealized = self.get_pl()
            current_holding = self.direction
            observation = self.data.iloc[self.current_step][['pips_delta', 'tl_slope']].values 
            observation = np.append(observation, [current_unrealized, current_holding])
            observation = self.deNaN(observation)
        else:
            self.done = True
            observation = self.deNaN([np.nan, np.nan, np.nan, np.nan])
            #print('reached end of data')
            #observation = self.reset()
            #return obs
        return observation


    def reward_function(self, pips_earned, time_underwater, total_ticks, gamma, exploration_bonus, maximal_drawdown_penalty):
        """
        This function returns a reward for a trade, based on the money earned, the time the trade was underwater,
        and the maximal drawdown of the trade.
        """
        #print('hello from reward_function().  len(self.data): ', len(self.data))
        # Calculate the profit component
        profit_reward = pips_earned

        # Calculate the time underwater penalty
        underwater_fraction = time_underwater/total_ticks if total_ticks>0 else time_underwater
        # Calculate time_penalty using underwater_fraction
        time_penalty = underwater_fraction * (gamma ** time_underwater)

        # Calculate maximal_drawdown only if pl_high_water_mark is not zero
        if self.pl_high_water_mark == 0:
            self.pl_high_water_mark = 0.00000001
        maximal_drawdown = (self.pl_high_water_mark - self.pl_low_water_mark) / self.pl_high_water_mark

        # Calculate the maximal drawdown penalty
        maximal_drawdown_penalty = maximal_drawdown_penalty * maximal_drawdown

        # Calculate the exploration bonus
        exploration_reward = exploration_bonus

        # Combine the components with discounting
        reward = profit_reward - time_penalty - maximal_drawdown_penalty + exploration_reward

        # Ensure that the rewards are not too sparse by capping them at a minimum value.

        reward = max(reward, 0.01)
        return pips_earned
        #return reward

    def reward_function5(self, pips_earned, above_water_fraction):
            # Use root logger to check if the configuration is affecting the method
            logging.info('reward_function5(), executing')
            
            risk_adj_reward = pips_earned * above_water_fraction
            reward = risk_adj_reward
            logging.info('reward_function5(), reward = %s', reward)
            return reward

    
    def get_pl(self):
        denominator = 10 ** float(self.pip_location)
        pl_pips = (self.current_price - self.open_price) / denominator
        return pl_pips

    def update(self, current_step, current_timestamp, current_price):
        # skip update if no position, i.e. volume is zero
        if(self.volume==0):
            return
        #set direction by volume
        self.direction = 1 if self.volume>0 else -1
        self.current_price = current_price
        self.p_l = self.get_pl()
        self.duration = current_timestamp - self.open_time
        # update ticks underwater
        if self.p_l < 0:
            self.ticks_underwater += 1
        # update ticks abovewater
        if self.p_l > 0:
            self.ticks_abovewater += 1

        # update total ticks
        self.total_ticks += 1

        # underwater_fraction
        self.underwater_fraction = self.ticks_underwater / self.total_ticks

        # above_water_fraction
        self.above_water_fraction = self.ticks_abovewater / self.total_ticks

        # hwm
        if self.p_l > self.hwm:
            self.hwm = self.p_l

        # append to equity curve
        self.equity_curve.append(self.equity_curve[-1] + self.p_l)

        # drawdown
        #drawdown = self.hwm - self.p_l
        drawdown = self.calculate_drawdown()
        if drawdown < self.mdd: # assumes dd is negative, need to check it
            self.mdd = drawdown

        # lwm
        if self.lwm is None or self.p_l < self.lwm:
            self.lwm = self.p_l

        # update metrics
        if(self.underwater_fraction ==0): #need to reexamine
            self.underwater_fraction = 1
        self.profit_over_underwater_fraction = self.p_l / self.underwater_fraction
        self.profit_over_mdd = self.p_l / self.mdd
        self.profit_over_underwater_fraction_and_mdd = self.profit_over_underwater_fraction / self.mdd


    def close_position(self, current_price, current_time, current_step):
        self.close_time = current_time
        self.close_price = current_price
        # update pos.
        self.update(current_step, current_time, current_price)
        jsn = self.get_position_json()
        self.reset_position()
        return jsn

    def reset_position(self):
        self.volume = 0.0
        self.equity_curve = [0]
        self.mdd = 0.0
        self.p_l = 0.0
        #print('Position reset')

    def calculate_drawdown(self):
        peak = max(self.equity_curve) if max(self.equity_curve)!=0 else 0.000001  #divide by zero
        trough = min(self.equity_curve)
        return (trough - peak) / peak

    def get_position_string(self):
        s = 'Buy' if self.direction==1 else 'Sell'
        s+=':\n'
        s+='open_price: ' + str(self.open_price) + '\n'
        s+='close_price: ' + str(self.close_price) + '\n'
        s+='self.p_l: ' + str(self.p_l) + '\n'
        s+='open_time: '+ str(self.open_time) + '\n'
        s+='close_time: '+ str(self.close_time) + '\n'
        s+='duration: '+ str(self.duration) + '\n'
        return s

    def get_position_json(self):
        position_data = {
            'direction': self.direction,
            'open_price': self.open_price,
            'close_price': self.close_price,
            'p_l': self.p_l,
            'open_time': self.open_time.strftime('%Y-%m-%d %H:%M:%S'),  # Convert to string
            'close_time': self.close_time.strftime('%Y-%m-%d %H:%M:%S'),  # Convert to string
            'duration': str(self.duration),  # Convert to string
            'above_water_fraction': str(self.above_water_fraction)
        }
        return json.dumps(position_data)

    def print_position(self):
        print(self.get_position_string())