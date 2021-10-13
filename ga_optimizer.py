import numpy as np
from random import random, uniform
from copy import deepcopy
from scipy.sparse.construct import rand
from sklearn.metrics import accuracy_score

class GAOptimizer:
    def __init__(self, n_weights, meta_model, X_train, y_train, X_valid, y_valid,
        n_generations=1000, pop_size=30, mutation_prob=0.1, crossover_prob=0.7, selection_type='tournment'):
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.selection_type = selection_type
        self.n_weights = n_weights
        self.population = []
        self.n_generations = n_generations
        self.meta_model = meta_model

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid


    def select(self, mating_pool):
        return self.tournment_selection(mating_pool)

    def tournment_selection(self, mating_pool):
        selected = []

        for i in range(self.pop_size):
            random_pos_1 = round(uniform(0, len(mating_pool)-1))
            random_pos_2 = round(uniform(0, len(mating_pool)-1))

            if mating_pool[random_pos_1]['fit'] >= mating_pool[random_pos_2]['fit']:
                selected.append(mating_pool[random_pos_1])
            else:
                selected.append(mating_pool[random_pos_2])

        return selected
    
    def mutate(self, chromossome):
        weight_change = uniform(0.3, 2.0) #neighbor_distance
        chosen_weight = round(uniform(0, len(chromossome['weights'])-1))

        new_weights = deepcopy(chromossome['weights'])
        new_weights[chosen_weight] = new_weights[chosen_weight] + weight_change
        normalized_weights = self.__normalize_weights(new_weights)

        return {
            'weights': normalized_weights,
            'fit': self.__evaluate_chromossome(normalized_weights)
        }

    def crossover(self, chromossome1, chromossome2):
        weights_len = len(chromossome1['weights'])
        random_pos = round(uniform(0, weights_len-1))
        chromossome1_weights, chromossome2_weights = chromossome1['weights'], chromossome2['weights']
        
        offspring1_weights = chromossome1_weights[:random_pos] + chromossome2_weights[random_pos:]
        offspring2_weights = chromossome2_weights[:random_pos] + chromossome1_weights[random_pos:]
        offspring1_weights = self.__normalize_weights(offspring1_weights)
        offspring2_weights = self.__normalize_weights(offspring2_weights)


        offspring1 = { 'weights': offspring1_weights, 'fit': self.__evaluate_chromossome(offspring1_weights) }
        offspring2 = { 'weights': offspring2_weights, 'fit': self.__evaluate_chromossome(offspring2_weights) }

        return offspring1, offspring2

    def optimize(self):
        self.population = self.__generate_population()

        for g in range(self.n_generations):
            mating_pool = []
            for p in range(self.pop_size):
                if random() <= self.mutation_prob:
                    mutant = self.mutate(self.population[p])
                    mating_pool.append(mutant)

                if random() <= self.crossover_prob:
                    random_chromossome_pos = round(uniform(0, self.pop_size-1))
                    offsprings = self.crossover(
                        self.population[p], self.population[random_chromossome_pos])
                    mating_pool += offsprings

            self.population = self.select(mating_pool)
                
            self.population.sort(key=lambda x: x['fit'], reverse=True)

            print(g)
            print([p['fit'] for p in self.population])
            
        return self.population[0]['weights']

    def __generate_population(self):
        new_population = []
        for i in range(self.pop_size):
            new_weights = self.__generate_weights(self.n_weights)
            new_population.append({
                'weights': new_weights,
                'fit': self.__evaluate_chromossome(new_weights)
            })

        return new_population

    def __evaluate_chromossome(self, weights):
        meta_model_cpy = deepcopy(self.meta_model)
        weighted_X_train = np.array(self.X_train).transpose() * np.array(weights)
        weighted_X_valid = np.array(self.X_valid).transpose() * np.array(weights)
        meta_model_cpy.fit(weighted_X_train, self.y_train)

        y_pred = meta_model_cpy.predict(weighted_X_valid)

        return accuracy_score(self.y_valid, y_pred)

    def __generate_weights(self, n_weights):
        weights = []
        new_weights = np.random.dirichlet(np.ones(n_weights),size=1)[0]

        for w in new_weights:
            weights.append(w)

        return weights
    
    def __normalize_weights(self, weights):
        total = sum(weights)
        normalized_weights = []

        for w in weights:
            normalized_weights.append(w / total)

        return normalized_weights