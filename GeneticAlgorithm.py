import jumanji
import jax.numpy as jnp
import jax
import random
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import os
from time import sleep

# Helper functions

def convert_state(state):
  '''
  Convert state into observation variable that can be passed into neural network
  PARAM -
  state: current game state
  RETURNS - observation variable
  '''
  grid = state.grid_padded.flatten().tolist()
  tetromino = state.tetromino_index.flatten().tolist()
  obs_variable = np.asarray(grid+tetromino)
  # print("Current input dimensions:")
  # print(np.shape(obs_variable))
  # print(obs_variable)
  return obs_variable

def unravel(state, action_index):
  array = np.array(state.action_mask)
  rotation, col_index = np.unravel_index(action_index, array.shape)
  action = [rotation, col_index]
  return action

class Population:

  def __init__(self, pop_size, per_mutation, per_crossover, per_elites):
    self.pop_size = pop_size # 30
    self.per_mutation = per_mutation # 0.4
    self.per_crossover = per_crossover # 0.4
    self.per_elites = per_elites # 0.2
    self.curr_gen = 1
    self.population = []
    self.elites = []

  def breed(self):
    '''
    Breeds initial population of solutions
    '''
    for i in range(self.pop_size):
      new_genome = Genome(122,64,64,32)
      self.population.append(new_genome)

  def parent_crossover(self,parent_a, parent_b):
    '''
    Creates another genome based off two parents where their weighting and biases mixed between 
    the two in a one point crossover. Returns offspring.
    '''
    offspring = Genome(122,64,64,32)
    # Get weight and biases of parents hidden layer
    weights1, biases1 = parent_a.model.layers[1].get_weights()
    weights2, biases2 = parent_b.model.layers[1].get_weights()
    weights_a, biases_a = parent_a.model.layers[2].get_weights()
    weights_b, biases_b = parent_b.model.layers[2].get_weights()
    # Swap weights between parents to create the offspring
    midpoint_w = len(weights1) // 2
    midpoint_b = len(biases1) // 2
    midpoint_w2 = len(weights_a) // 2
    midpoint_b2 = len(biases_a) // 2
    # Combine weights and biases from parents up to the midpoint
    offspring_weights1 = np.concatenate((weights1[:midpoint_w], weights2[midpoint_w:]), axis=0)
    offspring_biases1 = np.concatenate((biases1[:midpoint_b], biases2[midpoint_b:]), axis=0)
    offspring_weights2 = np.concatenate((weights_b[:midpoint_w2], weights_a[midpoint_w2:]), axis=0)
    offspring_biases2 = np.concatenate((biases_b[:midpoint_b2], biases_a[midpoint_b2:]), axis=0)
    # Set weights of offspring
    offspring.model.layers[1].set_weights([offspring_weights1, offspring_biases1])
    offspring.model.layers[2].set_weights([offspring_weights2, offspring_biases2])

    return offspring

  def mutation(self, curr_genome, mutation_rate):
    '''
    Mutates an agent where at each weight a dice is rolled whether or not the weighting changes
    '''
    # Mutates each layer individually
    model = curr_genome.model
    hidden_layer = model.layers[1]
    hidden_layer2 = model.layers[2]

    weights, biases = hidden_layer.get_weights()
    weights2, biases2 = hidden_layer2.get_weights()

    for i in range(len(weights)):
      for j in range(len(weights[i])):
        if np.random.rand() < mutation_rate:
          # Apply mutation to the weight
          # Can adjust the mutation strategy here
          weights[i][j] += np.random.normal(-0.1, 0.1)
    for i in range(len(weights2)):
      for j in range(len(weights2[i])):
        if np.random.rand() < mutation_rate:
          # Apply mutation to the weight
          # Can adjust the mutation strategy here
          weights2[i][j] += np.random.normal(-0.1, 0.1)

    curr_genome.model.layers[1].set_weights([weights, biases])
    curr_genome.model.layers[2].set_weights([weights2, biases2])


  def sort_population(self, fitness_scores):
    '''
    Sorts population given fitness score array
    '''
    # Sort in descending order
    sorted_indices = np.argsort(fitness_scores)[::-1]  
    self.population = [self.population[i] for i in sorted_indices]

  def choose_elites(self, elite_percentage):
    '''
    Chooses elite solutions of current generation AFTER SORTING THE POPULATION. 
    Will duplicate the elites.
    '''
    self.elites = []
    elite_count = int(elite_percentage * self.pop_size)

    for i in range(elite_count):
        self.elites.append(self.population[i])

  def roulette_wheel_selection(self, fitness_scores):
    '''
    SPIN THE WHEEL BABY!! Selects genomes with probability proprotional to their fitness score 
    for crossover.
    '''
    # Adds a small constant to fitness scores to allow agents with 0 fitness to still be chosen ensuring diversity
    scaled_fitness_scores = np.array(fitness_scores) + 0.01 
    # Get array of selection probability for each genome in population given array of fitness scores
    total = np.sum(scaled_fitness_scores)
    selection_probs = scaled_fitness_scores / total
    # Chooses genomes based on fitness scores. Chooses 40% of population.
    curr_pop_size = len(self.population)
    selected_indices = np.random.choice(curr_pop_size, size=int(curr_pop_size*self.per_crossover), p=selection_probs)

    selected_genomes = [self.population[i] for i in selected_indices]

    return selected_genomes

  def evolve(self):
    '''
    Selects which genome survives to next generation and which genome dies ＼（⊙ｏ⊙）／. Will use
    steady state evolution (making all new offpsrings will massively slow down process)
    '''
    # Get stastistics
    max_fit, min_fit, median_fit, avg_fit, std_fit, total_fit, fitness_scores = self.fitness_stats()
    print("Stastistics done")
    print("Max fitness: {0}".format(max_fit))
    # print("Min fitness: {0}".format(min_fit))
    print("Median fitness: {0}".format(median_fit))
    print("Avg fitness: {0}".format(avg_fit))
    print("Std fitness: {0}".format(std_fit))
    # print("Total fitness: {0}".format(total_fit))

    # Sort population
    self.sort_population(fitness_scores)
    print("Sorting done")

    # Choose elites with a percentage of how much you keep. Current 20%
    self.choose_elites(self.per_elites)
    print("Elites done")

    # Chooses 40% of genomes based using roulette wheel selection for crossover.
    selected_genomes = self.roulette_wheel_selection(fitness_scores)
    print("Roulette done")

    # Crossover selected genomes
    offspring_count = int(len(self.population) * (self.per_crossover/2))
    print("Current offspring count {0}".format(offspring_count))
    for _ in range(offspring_count):
      parent_a = np.random.choice(selected_genomes)
      parent_b = np.random.choice(selected_genomes)
      offspring = self.parent_crossover(parent_a, parent_b)
      index_to_replace = np.random.choice(range(len(self.elites), len(self.population)))
      self.population[index_to_replace] = offspring
    print("Crossover done")
    
    # Mutates 40% of population with probabiliy 0.1
    mutate_count = int(len(self.population) * self.per_mutation)
    print("Current mutate count {0}".format(mutate_count))
    for _ in range(mutate_count):
      index_to_mutate = np.random.choice(range(len(self.elites), len(self.population)))
      mutate_genome = self.population[index_to_mutate]
      self.mutation(mutate_genome, 0.001)
    print("Mutation done")


    # for genome in self.population:
    #   print(genome.fitness)

    self.curr_gen += 1

    return max_fit, min_fit, median_fit, avg_fit, std_fit, total_fit


  def fitness_stats(self):
    '''
    Get stastistics of the current generation of population. Need those analysis marks!
    '''
    fitness_scores = []
    for genome in self.population:
       fitness_scores.append(genome.fitness)
    
    max_fit = np.max(fitness_scores)
    min_fit = np.min(fitness_scores)
    median_fit = np.median(fitness_scores)
    avg_fit = np.sum(fitness_scores) / len(fitness_scores)
    std_fit = np.std(fitness_scores)
    total_fit = np.sum(fitness_scores)

    return max_fit, min_fit, median_fit, avg_fit, std_fit, total_fit, fitness_scores

  def run_gen(self):
    '''
    Run experience for all genomes in the population to calculate their fitness scores.
    '''
    print("Current population size: {0}".format(len(self.population)))
    counter = 0

    for curr_genome in self.population:
      curr_genome.fitness = 0
      # Instantiate tetris environment using registry
      env = jumanji.make('Tetris-v0', num_rows = 8, num_cols = 8, time_limit = 500)
      key = jax.random.PRNGKey(1)
      state, timestep = jax.jit(env.reset)(key)

      # Run a whole episode for a genome
      while True:
        action = curr_genome.policy(state)
        if action is False:
          break
        next_state, next_timestep = jax.jit(env.step)(state, action)
        curr_genome.fitness += next_state.reward
        state = next_state

        #env.render(state)

      counter += 1
      print("Generation {0} Genome {1} has been completed with fitness {2}".format(self.curr_gen, counter, curr_genome.fitness))


class Genome:

  def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
    self.model = self.build_model(input_size, hidden_size1, hidden_size2, output_size)
    self.fitness = 0

  def build_model(self, input_size, hidden_size1, hidden_size2, output_size):
    '''
    Builds neural network for a single genome. Hidden layers have random weights and biases.
    '''
    # 128, 64, 40 input, hidden,output
    # 256, 64, 64, 40 new

    # Vary the weights and biases for each genome
    kernel_init = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
    bias_init = initializer = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
    model = Sequential()
    # Adds fully connected layer with 128 units and uses rectified linear unit activation function. he_uniform initliazes weight of layer
    model.add(Dense(256, input_dim = input_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(hidden_size1, activation='relu', use_bias=True, bias_initializer = bias_init, kernel_initializer= kernel_init))
    model.add(Dense(hidden_size2, activation = 'relu', use_bias=True, bias_initializer = bias_init, kernel_initializer= kernel_init))
    # 40 actions so 40 different outputs (32 now)
    model.add(Dense(output_size, activation='linear', kernel_initializer='he_uniform'))
    # opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer = 'adam')
    return model

  def policy(self, state):
    '''
    Chooses action which has the highest output and is HOPEFULLY not illegal.
    '''
    # Matches state into 170(for 10x10), 122(8x8), array
    state_array = convert_state(state)
    # Convert to into tensorflow tensor
    state_input = tf.convert_to_tensor(state_array[None, :], dtype=tf.float32)
    # Grabs Q values for all possible actions in current state
    action_q = np.array(self.model(state_input))
    # Allows only legal moves
    action_mask = np.array(state.action_mask).flatten()
    # Checks if terminal state is reached when there are no more legal moves
    terminal = np.where(action_mask)[0]
    if len(terminal) == 0:
      return False
    else:
      # Legal moves will have a value of 1 while illegal moves have a value of 0
      locater = action_mask.astype(int)
      # Ensures illegal moves remain illegal
      max_q = np.absolute(action_q).max()
      action_q[:,locater == 0] -= (max_q + 9999999999)
      # Grabs the index of the action and converts it to suitable type
      action_index = np.argmax(action_q)
      action = unravel(state, action_index)
      return action
    


# Population parameters(population, percentage of mutation, percentage of crossover, percentage of elites)
# pop = 8
# p_m = 0.5 # 4
# p_c = 0.25 # 2
# p_e = 0.25 # 2

pop = 20
p_m = 0.4 # 8
p_c = 0.4 # 8
p_e = 0.2 # 4


max_gen = 200
interval = 5
Results = []
best_fitness = -10000

# Specify the directory to save the files
save_dir = "saved_models_and_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


print(tf.config.list_physical_devices('GPU'))

print("Starting population")
Big_poppa = Population(pop, p_m, p_c, p_e)
Big_poppa.breed()

for generation in range(max_gen):
    print("Generation:", generation+1)
    
    Big_poppa.run_gen()
    max_fit, min_fit, median_fit, avg_fit, std_fit, total_fit = Big_poppa.evolve()
    Results.append((max_fit, min_fit, median_fit, avg_fit, std_fit, total_fit))

    # Update best genome
    if max_fit > best_fitness:
        best_fitness = max_fit
        best_genome = Big_poppa.population[0]  # Assuming elites contain the best genomes
        
    
    print("Current population size:", len(Big_poppa.population))
    print("----")

    if generation % interval == 0:
        print("Saving results and best genome...")
        # Save results to a file or data structure
        # Save results to a file (e.g., CSV)
        results_filename = os.path.join(save_dir, f"results_generation_{generation}.csv")
        with open(results_filename, "w") as results_file:
          results_file.write("Generation,Max_Fitness,Min_Fitness,Median_Fitness,Avg_Fitness,Std_Fitness,Total_Fitness\n")
          for gen, (fit, min_fit, median_fit, avg_fit, std_fit, total_fit) in enumerate(Results, start=1):
            results_file.write(f"{gen},{fit},{min_fit},{median_fit},{avg_fit},{std_fit},{total_fit}\n")
        # Save best genome to a file or data structure
        # Save best genome as an HDF5 model file
        best_genome_filename = os.path.join(save_dir, f"best_genome_generation_{generation}.h5")
        best_genome.model.save(best_genome_filename)

print("Done")