# Code for evaluating a single genome
def evaluate_genome_new(queue, data_tuple, local_simulation_vars): # input_tuple: (neat_df, network)
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
            #print('reached end of data, last/nan obs sent: ', observation)
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

    def mystep(action, local_simulation_vars):
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
            #
            #reward = 0.001
            risk_adj_reward = pips_earned * float(local_simulation_vars['above_water_fraction'])
            reward = risk_adj_reward
            return reward

        # update done
        local_simulation_vars['done'] = local_simulation_vars['current_step'] >= len(data)
        
        # if episode NOT done
        if not local_simulation_vars['done']:          
            local_simulation_vars['current_timestamp'] = data['time'].iloc[local_simulation_vars['current_step']] #data.loc[local_simulation_vars['current_step'], 'time']
            #local_simulation_vars['current_step']
            #logging.info('current_step %s',local_simulation_vars['current_step'])
            local_simulation_vars['current_price'] = data['bid_c'].iloc[local_simulation_vars['current_step']]
            #logging.info('current_price %s',local_simulation_vars['current_price'])
            #reward = 0.0001  
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
                    reward = reward_function(local_simulation_vars['pips_earned'], local_simulation_vars['above_water_fraction'])
                    logging.info('on close reward = %s', reward)
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
        next_observation, reward, local_simulation_vars['done'], info = mystep(action, local_simulation_vars)
        total_reward += reward
        
    # end of simulation, zero out non-traders, and add to the queue
    # Keep track of the top genomes and their rewards
    genome_info = {'total_reward': total_reward, 'genome_id': genome_id, 'trades_list': trades_list}  # Replace current_genome_id with the actual ID of the genome
    top_genomes.append(genome_info)
    top_genomes = sorted(top_genomes, key=lambda x: x['total_reward'], reverse=True)[:10]  # Keep only the top ten genomes
    #logging.info('genome_info: ', genome_info)
    #print('genome_info2: ', genome_info)
    # write top genomes to file?
    if len(trades_list)>0:
        queue.put((genome_id, total_reward)) #, trades_list))
    else:
        #print('empty trades list - zeroing reward ')
        total_reward = 0.0
    
    return total_reward
# end evaluate_genome_new()
