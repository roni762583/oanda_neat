import visualize
import numpy as np

class NEATAgent:
    ''' Create a NEATAgent class that interacts with the custom TradingEnvironment '''
    def __init__(self, trading_env, genome, config):
        self.trading_env = trading_env
        self.genome = genome
        self.config = config
        self.net = None  # Initialize the neural network as None

    def initialize_network(self):
        #pass
        print('hello from NEATAgent.initialize_network()')
        #print('initialize_network only sets env net to the agent net')
        #self.net = FeedForwardNetwork.create(self.genome, self.config)
        #self.trading_env.net = self.net

    def set_network(self, input_net):
        self.net = input_net
        self.trading_env.net = self.net
        #print('hello from set_network()')
        '''gets net object from parameter and sets self.net, as well as env copy of net
        '''

    def simulate_trading(self, queue):
        if self.net is None:
            self.initialize_network()  # Initialize the network if it hasn't been done already
            print('why m i calling dead function???')
        total_reward = 0.0
        #print('b4 reset self.trading_env.done',self.trading_env.done)
        next_observation = self.trading_env._next_observation() #self.trading_env.reset() # reset returns initial observation
        #self.trading_env.done = False
        print('first.next_observation: ', next_observation)

        while not self.trading_env.done:
            #print('yay not done')
            output = self.net.activate(next_observation)
            #print('output: ', output)
            action = self.get_action(output)
            #print('action: ', action)
            next_observation, reward, self.trading_env.done, info = self.trading_env.step(action)
            if reward!=1.0:
                print('next_observation: ', next_observation, ', row: ', self.trading_env.current_step, ', reward: ', reward)
            
            total_reward += reward
        #total_reward=0.298 this works!
        # add to the queue
        queue.put(total_reward)#total_reward)
    
        return total_reward


    # Define the custom softmax activation function
    def softmax_activation(self, x):
        """Custom softmax activation function."""
        #print('hello from softmax_activation()')
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_action(self, output):
        """Get the action with the highest probability."""
        #print('hello from get_action()')
        output = self.softmax_activation(output)
        action = np.argmax(output)
        #print('action', action)
        return action

    def visualize_genome(self, show=True):
        # Visualize the NEAT genome
        node_names = {-1: 'Input1', -2: 'Input2', 0: 'Output'}
        visualize.draw_net(self.config, self.genome, show, node_names=node_names)

    def plot_statistics(self, stats, ylog=False, view=True):
        # Plot NEAT statistics
        visualize.plot_stats(stats, ylog=ylog, view=view)

    def plot_species(self, stats, view=True):
        # Plot NEAT species
        visualize.plot_species(stats, view=view)

