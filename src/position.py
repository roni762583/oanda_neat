from config.experiment_config import spread
import json

class Position:
        def __init__(self, current_price, open_time, pip_location, volume=0):
            cost = -1*spread
            # negative volume represents a short trade
            self.volume = volume
            self.current_price = current_price
            self.open_time = open_time
            self.open_price = 0.0
            self.pip_location = pip_location
            self.p_l = cost
            self.close_time = None
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
            self.reset()
            return jsn

        def reset(self):
            self.volume = 0
            self.equity_curve = [0]
            self.mdd = 0
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

        ### END CLASS Position