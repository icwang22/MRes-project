# -*- coding: utf-8 -*-
"""
Created on Tue Dec 8 20:06:43 2015

@author: Taran Driver

To do:
- use symmetry of circle/rhombus to quicken up jackknife program etc.
- optimise jackknife parameters
- negate need for a/c peak filtering in peak reading programs
- allow making jackknife map
"""

# Calculate significances and plot them as scatter plots, scanning the number
# of scans

# All 'Resample' functions refer to jackknife resampling!

# For this library, you already have to be in the directory with the
# data for the specific peptide.

# To do:
# Si and Si2 can be a method of the cvMap, not the peak, if the pCovParam is
# m/z invariant

import os
import numpy as np
from scipy import stats
from scipy import interpolate
from scipy import ndimage
from pC2DMSUtils import maxIndices, varII, covXI, cutAC, saveSyxEinSum, scaleToPower, circList, clearCirc, bsResamps
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from scipy.optimize import differential_evolution, dual_annealing, minimize, Bounds
import concurrent.futures
import pyscipopt as scip
from pyswarm import pso
import pyswarms as ps
import optuna
import cloudpickle
import dill
import cma
'''import nevergrad as ng'''

class Scan:
    def __init__(self, scanFolder, AGCtarget=100):
        self.scanFolder = scanFolder
        self.AGCtarget = AGCtarget
        self.normFactor = AGCtarget / 10000.

        self.scanList = self.normFactor * np.load(scanFolder + '/array.npy', allow_pickle=True, mmap_mode='r')[1:]

        params = np.load(scanFolder + '/array_parameters.npy', allow_pickle=True)
        self.sliceSize = params[3]
        self.minMZ = params[5]  # using this currently means index2mz (and
        # equally, mz2index) gives an
        # m/z value which is 1*self.sliceSize lower than the mz value that is
        # provided for the reading in the text file, but for the charge states
        # and scan mode (Turbo) that we have been working with this has been
        # very useful because it compensates for the fact that because of the
        # isotopic envelope the top of the Turbo peak (which doesn't resolve
        # the isotopic envelope) is shifted up in m/z from the monoisotopic m/z
        self.maxMZ = params[6]
        # The text file that gets read in reports a min mz and max mz and then
        # gives intensity readings at each m/z value. The min m/z value at
        # which is gives an intensity reading for is one sliceSize higher than
        # the reported min m/z, the max m/z value that it gives an intensity
        # reading for is the same as the reported max m/z.
        self.fullNumScans = int(params[1])

    def tic(self):
        return self.scanList.sum(axis=1)

    def aveTIC(self):
        'Mean-averaged total ion count across all scans'
        return self.totIonCount().sum() / self.fullNumScans

    def ionCount(self, centralMZ, width, numScans='all', returnBounds=False):
        """returns ion count across a certain m/z range defined by central
        m/z and width. Output is 1D array of count for each scan"""
        binsOut = np.ceil(float(width) / (self.sliceSize * 2.))  # so as to err on
        # the side of the range being too large
        centralIndex = self.mz2index(centralMZ)

        fromIndex = centralIndex - binsOut
        toIndex = centralIndex + binsOut

        x = self.fullNumScans if numScans == 'all' else numScans

        fromMZ = round(self.index2mz(fromIndex), 2)
        toMZ = round(self.index2mz(toIndex), 2)

        if returnBounds:
            return self.scanList[:x, fromIndex:toIndex + 1].sum(axis=1), fromMZ, toMZ
        else:
            return self.scanList[:x, fromIndex:toIndex + 1].sum(axis=1)

    def index2mz(self, index):
        """Provides the m/z value relating to the specified index for the
        relevant scan dataset. index need not be an integer value."""
        return self.minMZ + index * self.sliceSize

    def mz2index(self, mz):
        """Provides the closest (rounded) integer m/z slice index relating to
        the m/z value for the relevant scan dataset."""
        if mz < self.minMZ or mz > self.maxMZ:
            raise ValueError('m/z value outside range of m/z values for this scan')
        if mz % self.sliceSize < self.sliceSize / 2:
            return int((mz - self.minMZ) / self.sliceSize)
        else:
            return int((mz - self.minMZ) / self.sliceSize + 1)

    def weightFunc(self, weights=None):
        """
        Calculate weighted sum of intensities across all scans. If weights is None,
        a uniform weighting is used.

        Parameters:
        weights (array-like): 1D array of weights for each m/z slice. If shorter than
            the number of rows in self.scanList, a group of m/z values will correspond
            to the same weight.

        Returns:
        array-like: 1D array of weighted sums of intensities for each m/z slice.
        """
        if weights is None:
            weights = np.ones(self.scanList.shape[1])
        elif len(weights) < self.scanList.shape[1]:
            weights = np.repeat(weights, self.scanList.shape[1] // len(weights) + 1)[:self.scanList.shape[1]]
        return np.einsum('ij,j->i', self.scanList, weights)

    def generateInitialWeights(self, range_size=300):
        # Generate an initial guess of the weights
        loc = (self.maxMZ + self.minMZ) / 2
        scale = (self.maxMZ - self.minMZ) / 10
        size = self.scanList.shape[1] // range_size
        # compute weights using inverse Gaussian distribution
        weights = stats.invgauss.rvs(loc=loc, scale=scale, size=size, mu=1)
        weights = weights / weights.mean()
        return weights
    
    def generateInitialWeightsEqual(self, range_size=300):
        size = self.scanList.shape[1] // range_size
        # Generate an initial guess of the weights
        weights = np.ones(size)
        return weights

    def optimizeWeights(self, indexList, loss_function, scan_dir, weights=None, range_size=100, optimizer_path='optimizer_bayopt.dill', iterations_per_run=1):
        """Optimizes weights using Bayesian Optimization algorithm."""

        # Generate initial guess of weights
        if weights is None:
            weights = self.generateInitialWeights(range_size)

        # Initial loss evaluation
        x0 = weights
        y0 = -loss_function(scan_dir, x0, indexList)
        y0 = np.squeeze(y0)  # remove any singleton dimensions

        kernel = ConstantKernel(constant_value_bounds=(1e-12, 1e3)) + Matern(length_scale_bounds=(1e-12, 1e3), nu=1.5)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=50
        )
        X0 = x0.reshape(-1, 1).T
        y0 = y0.reshape(-1)  # Reshape y0 to have shape (n_samples,)
        gpr.fit(X0, y0)

        bounds = [(0, 2) for _ in range(len(weights))]

        # Define optimizer
        optimizer = BayesianOptimization(
            f=lambda **weights: -loss_function(scan_dir, np.array(list(weights.values())), indexList),
            pbounds=dict(zip([f'w_{i}' for i in range(len(weights))], bounds)),
            verbose=2,
            random_state=42
        )

        # Update optimizer with surrogate model
        optimizer._gp = gpr

        # Load optimizer from disk if it exists
        loaded_optimizer = None
        if os.path.exists(optimizer_path):
            with open(optimizer_path, 'rb') as f:
                loaded_optimizer = dill.load(f)

        if loaded_optimizer is not None:
            # Continue optimization using the existing optimizer
            loaded_optimizer.maximize(init_points=0, n_iter=iterations_per_run)

            optimizer = loaded_optimizer

            # Save the combined optimizer using dill
            with open(optimizer_path, 'wb') as f:
                dill.dump(optimizer, f)
        else:
            # Run optimization
            optimizer.maximize(init_points=10, n_iter=iterations_per_run)

            # Save the optimizer using dill
            with open(optimizer_path, 'wb') as f:
                dill.dump(optimizer, f)

        # Return optimized weights
        optimized_weights = np.array([optimizer.max['params'][f'w_{i}'] for i in range(len(weights))])
        optimized_loss = -optimizer.max['target']

        return optimized_weights, optimized_loss
    
    
    def optimizeWeightsTPE(self, indexList, loss_function, scan_dir, weights=None, range_size=100, optimizer_path = "optimizer_tpe.dill"):
        """Optimizes weights using Tree-structured Parzen Estimators (TPE) algorithm with Optuna."""

        # Generate initial guess of weights
        if weights is None:
            weights = self.generateInitialWeights(range_size)

        # Define bounds for weights"""Optimizes weights using Tree-structured Parzen Estimators (TPE) algorithm with Optuna."""

        # Generate initial guess of weights
        if weights is None:
            weights = self.generateInitialWeights(range_size)

        # Define bounds for weights
        bounds = [(0, 2) for _ in range(len(weights))]

        # Define the objective function for optimization
        def objective_function(trial):
            trial_weights = [trial.suggest_float(f'w_{i}', 0, 2) for i in range(len(weights))]
            return -loss_function(scan_dir, np.array(trial_weights), indexList)

        # Load optimizer from disk if it exists
        loaded_optimizer = None
        if os.path.exists(optimizer_path):
            with open(optimizer_path, "rb") as file:
                loaded_optimizer = dill.load(file)

        if loaded_optimizer is not None:
            # Run optimization using the existing optimizer
            trial = loaded_optimizer.ask()
            loaded_optimizer.tell(trial, objective_function(trial))
            # Save the optimizer after each trial
            with open(optimizer_path, "wb") as file:
                dill.dump(loaded_optimizer, file)
        else:
            # Create a new optimizer using Optuna
            loaded_optimizer = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=1,
                    n_ei_candidates=24,
                    seed=42
                )
            )
            # Run optimization
            trial = loaded_optimizer.ask()
            loaded_optimizer.tell(trial, objective_function(trial))

        # Save the optimizer
        with open(optimizer_path, "wb") as file:
            dill.dump(loaded_optimizer, file)

        # Get the best weights and loss from the optimizer
        optimized_weights = [loaded_optimizer.best_params[f'w_{i}'] for i in range(len(weights))]
        optimized_loss = -loaded_optimizer.best_value

        return optimized_weights, optimized_loss
        bounds = [(0, 2) for _ in range(len(weights))]

        # Define the objective function for optimization
        def objective_function(trial):
            trial_weights = [trial.suggest_float(f'w_{i}', 0, 2) for i in range(len(weights))]
            return -loss_function(scan_dir, np.array(trial_weights), indexList)

        # Load optimizer from disk if it exists
        optimizer_path = "optimizer_tpe.dill"
        loaded_optimizer = None
        if os.path.exists(optimizer_path):
            with open(optimizer_path, "rb") as file:
                loaded_optimizer = dill.load(file)

        if loaded_optimizer is not None:
            # Run optimization using the existing optimizer
            trial = loaded_optimizer.ask()
            loaded_optimizer.tell(trial, objective_function(trial))
            # Save the optimizer after each trial
            with open(optimizer_path, "wb") as file:
                dill.dump(loaded_optimizer, file)
        else:
            # Create a new optimizer using Optuna
            loaded_optimizer = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=1,
                    n_ei_candidates=24,
                    seed=42
                )
            )
            trial = loaded_optimizer.ask()
            loaded_optimizer.tell(trial, objective_function(trial))

        # Save the optimizer
        with open(optimizer_path, "wb") as file:
            dill.dump(loaded_optimizer, file)

        # Get the best weights and loss from the optimizer
        optimized_weights = [loaded_optimizer.best_params[f'w_{i}'] for i in range(len(weights))]
        optimized_loss = -loaded_optimizer.best_value

        return optimized_weights, optimized_loss
    
    def optimizeWeightsPSO(self, indexList, loss_function, scan_dir, weights=None, range_size=30, iterations_per_run=1):
        """Optimizes weights using the Particle Swarm Optimization (PSO) algorithm."""
        
        def save_optimizer(optimizer, filename):
            """Save the optimizer's relevant attributes."""
            optimizer_info = {
                'n_particles': optimizer.n_particles,
                'dimensions': optimizer.dimensions,
                'options': optimizer.options,
                'bounds': optimizer.bounds,
                'best_pos': optimizer.swarm.best_pos,
                'best_cost': optimizer.swarm.best_cost
            }
            with open(filename, 'wb') as file:
                dill.dump(optimizer_info, file)

        def load_optimizer(filename):
            """Load the optimizer from the saved file."""
            with open(filename, 'rb') as file:
                optimizer_info = dill.load(file)
            n_particles = optimizer_info['n_particles']
            dimensions = optimizer_info['dimensions']
            options = optimizer_info['options']
            bounds = optimizer_info['bounds']
            optimizer = ps.single.GlobalBestPSO(
                n_particles=n_particles,
                dimensions=dimensions,
                options=options,
                bounds=bounds
            )
            optimizer.swarm.best_pos = optimizer_info['best_pos']
            optimizer.swarm.best_cost = optimizer_info['best_cost']
            return optimizer

        # Generate initial guess of weights
        if weights is None:
            weights = self.generateInitialWeights(range_size)

        # Define bounds for weights
        bounds = [(0, 2) for _ in range(len(weights))]

        # Transform bounds into the expected format
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        bounds_tuple = (lower_bounds, upper_bounds)

        # Define the objective function for optimization
        def objective_function(x):
            return -loss_function(scan_dir, x, indexList)

        def step(optimizer, loss_function, scan_dir, indexList):
            """Perform a single iteration (step) of Particle Swarm Optimization (PSO)."""
            current_costs = []

            for i in range(optimizer.n_particles):
                particle = optimizer.swarm.position[i]
                velocity = optimizer.swarm.velocity[i]
                pbest = optimizer.swarm.pbest_pos[i]
                best_global = optimizer.swarm.best_pos

                # Update particle velocity
                r1 = np.random.uniform(size=optimizer.dimensions)
                r2 = np.random.uniform(size=optimizer.dimensions)
                cognitive_component = optimizer.options['c1'] * r1 * (pbest - particle)
                social_component = optimizer.options['c2'] * r2 * (best_global - particle)
                inertia_weight = np.random.uniform(0.4, 0.9)
                velocity = inertia_weight * velocity + cognitive_component + social_component

                # Update particle position with constraint handling
                particle = np.clip(particle + velocity, optimizer.bounds[0], optimizer.bounds[1])

                # Update particle position and velocity in optimizer swarm
                optimizer.swarm.position[i] = particle
                optimizer.swarm.velocity[i] = velocity

                # Evaluate particle fitness (cost)
                weights_dict = {f'weight_{idx}': weight for idx, weight in enumerate(particle)}
                weights_array = np.array([weight for weight in weights_dict.values()])
                cost = loss_function(scan_dir, weights_array, indexList)
                current_costs.append(cost)

                # Update personal best (pbest) if improved
                if cost < optimizer.swarm.pbest_cost[i]:  # Minimize the cost here
                    optimizer.swarm.pbest_pos[i] = particle
                    optimizer.swarm.pbest_cost[i] = cost

                    # Update global best (gbest) if improved
                    if cost < optimizer.swarm.best_cost:  # Minimize the cost here
                        optimizer.swarm.best_pos = particle
                        optimizer.swarm.best_cost = cost

            optimizer.swarm.current_cost = np.array(current_costs)  # Update current_cost array

            return optimizer

        optimizer_path = "optimizer_pso.dill"
    
        if os.path.exists(optimizer_path):
            optimizer = load_optimizer(optimizer_path)
        else:
            options = {'c1': 2.0, 'c2': 2.0, 'w': 0.5}
            optimizer = ps.single.GlobalBestPSO(
                n_particles=1,
                dimensions=len(weights),
                options=options,
                bounds=bounds_tuple
            )
            optimizer.optimize(objective_function, iters=1)

        # Perform optimization using the chosen optimizer
        for _ in range(iterations_per_run):
            optimizer = step(optimizer, loss_function, scan_dir, indexList)

        # Save the optimizer's relevant attributes
        save_optimizer(optimizer, optimizer_path)

        # Return optimized weights
        optimized_weights = optimizer.swarm.best_pos
        optimized_loss = optimizer.swarm.best_cost

        return optimized_weights, optimized_loss

    
    def optimizeWeightsCMA(self, indexList, loss_function, scan_dir, weights=None, range_size=30, optimizer_file = "optimizer_cma.dill", iterations_per_run=1):
        """Optimizes weights using CMA-ES algorithm."""

        # Generate initial guess of weights
        if weights is None:
            weights = self.generateInitialWeights(range_size)

        # Define the objective function for optimization
        def objective_function(x):
            return loss_function(scan_dir, x, indexList)

        # Check if the optimizer file exists and load the optimizer from disk
        if os.path.exists(optimizer_file):
            with open(optimizer_file, "rb") as file:
                optimizer = dill.load(file)
        else:
            # Create a new optimizer using CMA if not loaded from disk
            optimizer = cma.CMAEvolutionStrategy(x0=weights, sigma0=0.1)

        # Number of iterations
        num_iterations = iterations_per_run

        for iteration in range(num_iterations):
            # Get the next set of candidate solutions (weights)
            candidates = optimizer.ask()

            # Evaluate the objective function for each candidate
            results = [objective_function(candidate) for candidate in candidates]

            # Update the optimizer with the results (tell method)
            optimizer.tell(candidates, results)

            # Save the updated optimizer to disk (overwrite the existing one)
            with open(optimizer_file, "wb") as file:
                dill.dump(optimizer, file)

            # Get the optimized weights
            result = optimizer.result  # Retrieve the result object containing the best solution
            optimized_weights = result.xbest
            optimized_loss = result.fbest

            print("Weights:", optimized_weights)
            print("Loss:", optimized_loss)

        return optimized_weights, optimized_loss
    
    '''def optimizeWeightsNg(self, indexList, loss_function, scan_dir, weights=None, range_size=30):

        # Generate initial guess of weights
        if weights is None:
            weights = self.generateInitialWeights(range_size)

        # Define bounds for weights
        bounds = [(0, 2) for _ in range(len(weights))]

        # Define the objective function for optimization
        def objective_function(x):
            return -loss_function(scan_dir, x, indexList)

        optimizer_file = "optimizer_ng.dill"

        # Check if the optimizer file exists and load the optimizer from disk
        if os.path.exists(optimizer_file):
            with open(optimizer_file, "rb") as file:
                optimizer = dill.load(file)
        else:
            # Create a new optimizer using nevergrad if not loaded from disk
            parametrization = ng.p.Array(init=weights, lower=0, upper=2)
            optimizer = ng.optimizers.NGOpt(parametrization=parametrization)

        # Perform optimization using the existing optimizer
        optimizer.maximize(init_points=0, n_iter=1, verbosity=1)

        # Save the updated optimizer to disk
        with open(optimizer_file, "wb") as file:
            dill.dump(optimizer, file)

        # Get the optimized weights
        optimized_weights = optimizer.provide_recommendation().value
        optimized_loss = -optimizer.provide_recommendation().loss

        print("Optimization using nevergrad:")
        print("Weights:", optimized_weights)
        print("Loss:", optimized_loss)

        return optimized_weights, optimized_loss'''
    
    def optimizeWeightsDifferential_evolution(self, indexList, loss_function, scan_dir, weights=None, range_size=30):
        """Optimizes weights using a combination of optimization algorithms."""

        # Generate initial guess of weights
        if weights is None:
            weights = self.generateInitialWeights(range_size)
        
        # Define bounds for weights
        bounds = [(0, 2) for _ in range(len(weights))]

        # Define the objective function for optimization
        def objective_function(x):
            return -loss_function(scan_dir, x, indexList)
        
        print("optimization using differential evolution:")

        # Perform optimization using differential evolution
        result_de = differential_evolution(objective_function, bounds)

        # Get the optimized weights
        optimized_weights_de = result_de.x
        optimized_loss_de = -result_de.fun
        print('weights: ', optimized_weights_de)
        print("Loss: ", optimized_loss_de)

        print("optimization using PySCIPOpt:")
        
        # Perform optimization using PySCIPOpt
        model_scp = scip.Model()
        weights_scp = [model_scp.addVar(lb=0, ub=2, name=f'w_{i}') for i in range(len(weights))]
        loss_scp = loss_function(scan_dir, np.array(weights_scp), indexList)
        model_scp.setObjective(-loss_scp)
        model_scp.optimize()
        optimized_weights_scp = np.array([model_scp.getVal(w) for w in weights_scp])
        optimized_loss_scp = -model_scp.getObjVal()
        print('weights: ', optimized_weights_scp)
        print("Loss: ", optimized_loss_scp)

        # Compare the optimized losses and choose the best
        optimized_weights_list = [optimized_weights_de, optimized_weights_scp]
        optimized_losses = [optimized_loss_de, optimized_loss_scp]
        best_index = np.argmax(optimized_losses)
        optimized_weights = optimized_weights_list[best_index]
        optimized_loss = optimized_losses[best_index]

        return optimized_weights, optimized_loss
    
    def optimizeWeightsDual_annealing(self, indexList, loss_function, scan_dir, weights=None, range_size=30):
        """Optimizes weights using the dual_annealing algorithm."""

        # Generate initial guess of weights
        if weights is None:
            weights = self.generateInitialWeights(range_size)

        # Define bounds for weights
        bounds = [(0, 2) for _ in range(len(weights))]

        # Define the objective function for optimization
        def objective_function(x):
            return -loss_function(scan_dir, x, indexList)

        # Perform optimization using dual_annealing
        result = dual_annealing(objective_function, bounds)

        # Get the optimized weights
        optimized_weights = result.x
        optimized_loss = -result.fun

        print("Optimization using dual_annealing:")
        print("Weights:", optimized_weights)
        print("Loss:", optimized_loss)

        return optimized_weights, optimized_loss
    
    
    def optimizeWeightsPSOFast(self, indexList, loss_function, scan_dir, weights=None, range_size=30):
        """Optimizes weights using the Particle Swarm Optimization (PSO) algorithm."""

        # Generate initial guess of weights
        if weights is None:
            weights = self.generateInitialWeights(range_size)

        # Define bounds for weights
        bounds = [(0, 2) for _ in range(len(weights))]

        # Define the objective function for optimization
        def objective_function(x):
            return -loss_function(scan_dir, x, indexList)

        # Perform optimization using PSO
        optimized_weights, optimized_loss = pso(objective_function, np.array(bounds)[:, 0], np.array(bounds)[:, 1])

        print("Optimization using PSO:")
        print("Weights:", optimized_weights)
        print("Loss:", optimized_loss)

        return optimized_weights, optimized_loss


    def oneD(self, numScans='all'):
        x = self.fullNumScans if numScans == 'all' else numScans
        return self.scanList[:x].sum(0)

    def plot1D(self, numScans='all'):
        'Plot the averaged 1D spectrum from array.npy'
        oneD = self.oneD(numScans=numScans)
        plt.plot(np.linspace(self.minMZ, self.maxMZ, self.scanList.shape[1], endpoint=True),
                 oneD / (np.nanmax(oneD) * 0.01))
        plt.xlabel('m/z')
        plt.ylabel('Relative abundance, %')


class Map:
    'Simple or partial covariance map'

    def __init__(self, scan, numScans='all'):
        self.scan = scan
        if numScans == 'all':
            self.numScans = self.scan.fullNumScans
        else:
            self.numScans = numScans
        self.build()

    def syx(self):
        'Syx is attribute, syx is method'
        try:
            return self.Syx
        except:
            syxPath = self.scan.scanFolder + \
                      '/Syx_' + str(self.numScans) + '_scans.npy'

            if os.path.isfile(syxPath):
                return self.scan.normFactor ** 2 * np.load(syxPath, allow_pickle=True)
            else:
                print('Syx not saved for this map, beginning calculation with saveSyxEinSum...')
                saveSyxEinSum(self.scan.scanFolder, numScans=self.numScans)
                return self.scan.normFactor ** 2 * np.load(syxPath, allow_pickle=True)

    def loadSyx(self):
        'Syx is attribute, syx is method'
        self.Syx = self.syx()

    def plot(self, pixAcross, power, mapTitle='', save=False, figFileName='', \
             fullRange=True, minMZx=100, maxMZx=101, minMZy=100, maxMZy=101):

        cdict = {'red': ((0.0, 1.0, 1.0),
                         (0.16667, 0.0, 0.0),
                         (0.33333, 0.5, 0.5),
                         (0.5, 0.0, 0.0),
                         (0.66667, 1.0, 1.0),
                         (1, 1.0, 1.0)),

                 'green': ((0.0, 0.0, 0.0),
                           (0.16667, 0.0, 0.0),
                           (0.33333, 1.0, 1.0),
                           (0.5, 0.5, 0.5),
                           (0.66667, 1.0, 1.0),
                           (0.83333, 0.0, 0.0),
                           (1.0, 1.0, 1.0)),

                 'blue': ((0.0, 1.0, 1.0),
                          (0.33333, 1.0, 1.0),
                          (0.5, 0.0, 0.0),
                          (0.83333, 0.0, 0.0),
                          (1.0, 1.0, 1.0))}

        Alps1 = LinearSegmentedColormap('Alps1', cdict)

        toPlot = scaleToPower((cutAC(self.array, pixAcross)), power)  # these three operations

        if not fullRange:
            minIndexX = (minMZx - self.scan.minMZ) / self.scan.sliceSize
            maxIndexX = (maxMZx - self.scan.minMZ) / self.scan.sliceSize

            minIndexY = (minMZy - self.scan.minMZ) / self.scan.sliceSize
            maxIndexY = (maxMZy - self.scan.minMZ) / self.scan.sliceSize

            toPlot = toPlot[int(minIndexY):int(maxIndexY), \
                     int(minIndexX):int(maxIndexX)]

        v_max = np.nanmax(toPlot)
        v_min = -v_max

        plt.figure(figsize=((200 / 9), 20))

        fig1 = gridspec.GridSpec(36, 40)  # Set up GridSpec to allow custom
        # placement of figures
        cvMap = plt.subplot(fig1[0:29, 7:36])
        cvMap1 = cvMap.pcolorfast(toPlot, vmin=v_min, vmax=v_max, cmap=Alps1)

        plt.xlabel('Mass-to-Charge Ratio, Da/C')
        plt.ylabel('Mass-to-Charge Ratio, Da/C')
        plt.title(mapTitle, fontsize=14)

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

        cbar = plt.subplot(fig1[2:28, 38])
        plt.colorbar(cvMap1, cax=cbar)

        ints = self.scan.scanList[:self.numScans].sum(axis=0)
        mzs = np.linspace(self.scan.minMZ, self.scan.maxMZ, len(ints))

        if fullRange:
            oneDDataXHor = mzs
            oneDDataYHor = ints

            oneDDataXVer = mzs
            oneDDataYVer = ints

        else:

            minIndex1DHor = (minMZx - self.scan.minMZ) / self.scan.sliceSize
            maxIndex1DHor = (maxMZx - self.scan.minMZ) / self.scan.sliceSize
            oneDDataXHor = mzs[int(minIndex1DHor):int(maxIndex1DHor)]
            oneDDataYHor = ints[int(minIndex1DHor):int(maxIndex1DHor)]

            minIndex1DVer = (minMZy - self.scan.minMZ) / self.scan.sliceSize
            maxIndex1DVer = (maxMZy - self.scan.minMZ) / self.scan.sliceSize
            oneDDataXVer = mzs[int(minIndex1DVer):int(maxIndex1DVer)]
            oneDDataYVer = ints[int(minIndex1DVer):int(maxIndex1DVer)]

        oneDSpectrumHor = plt.subplot(fig1[30:36, 7:36])  # horizontal
        # 1D spectrum
        oneDSpectrumHor.plot(oneDDataXHor, oneDDataYHor)
        plt.axis('tight')
        plt.xlabel('Mass-to-Charge Ratio, Da/C')
        plt.ylabel('Normalised Signal Intensity')
        ax = plt.gca()
        ax.set_yticks([])
        ax.set_yticklabels([])

        oneDSpectrumVer = plt.subplot(fig1[0:29, 0:6])  # vertical
        # 1D spectrum
        oneDSpectrumVer.plot(oneDDataYVer, oneDDataXVer)
        plt.axis('tight')
        plt.xlabel('Normalised Signal Intensity')
        plt.ylabel('Mass-to-Charge Ratio, Da/C')
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_xticklabels([])

        # Set parameters ticks on the plots
        majorLocator = MultipleLocator(50)  # how far apart labelled ticks are
        minorLocator = MultipleLocator(5)  # how far apart unlabelled ticks are

        oneDSpectrumHor.xaxis.set_major_locator(majorLocator)
        oneDSpectrumHor.xaxis.set_minor_locator(minorLocator)

        oneDSpectrumVer.yaxis.set_major_locator(majorLocator)
        oneDSpectrumVer.yaxis.set_minor_locator(minorLocator)

        oneDSpectrumHor.tick_params(axis='both', length=6, width=1.5)
        oneDSpectrumHor.tick_params(which='minor', axis='both', length=4, \
                                    width=1.5)
        oneDSpectrumVer.tick_params(axis='both', length=6, width=1.5)
        oneDSpectrumVer.tick_params(which='minor', axis='both', length=4, \
                                    width=1.5)

        plt.show()

        if save:
            plt.savefig(figFileName)

        return

    def analyse(self, numFeats, clearRad=25, chemFilt=[], \
                chemFiltTol=2.0, shapeFilt=False, shFiltThr=-0.2, shFiltRef='map', \
                shFiltRad=15, breakTime=3600, \
                pixOut=15, comPixOut=3, cutPeaks=True, integThresh=0, numRays=100, \
                perimOffset=-0.5, pixWidth=1, sampling='jackknife', bsRS=None, \
                bsRS_diffs=None, saveAt=False, basePath=None, useT4R=False, \
                printAt=50):
        # last 13 kwargs (after break) are parameters for sampleFeats. bsRs is
        # the bootstrap resamples provided
        'Picks numFeats feats from map and then samples them using jackknife'

        indexList = self.topNfeats(numFeats, clearRad=clearRad, \
                                   chemFilt=chemFilt, chemFiltTol=chemFiltTol, shapeFilt=shapeFilt, \
                                   shFiltThr=shFiltThr, shFiltRef=shFiltRef, shFiltRad=shFiltRad, \
                                   breakTime=breakTime, boundFilt=True, boundLimit=pixOut, \
                                   returnDiscards=False)
        
        # boundFilt must be true with boundLimit as pixOut for sampleFeats
        # to not raise an exception

        return self.sampleFeats(indexList, pixOut=pixOut, \
                                comPixOut=comPixOut, cutPeaks=cutPeaks, integThresh=integThresh, \
                                numRays=numRays, perimOffset=perimOffset, pixWidth=pixWidth, \
                                sampling=sampling, bsRS=bsRS, bsRS_diffs=bsRS_diffs, saveAt=saveAt, \
                                basePath=basePath, useT4R=useT4R, printAt=printAt)

    def sampleFeats(self, indexList, pixOut=15, comPixOut=3,
                    cutPeaks=True, integThresh=0, numRays=100, perimOffset=-0.5,
                    pixWidth=1, sampling='jackknife', bsRS=None, bsRS_diffs=None,
                    useT4R=False, saveAt=False, basePath=None, printAt=50):
        'Returns m/z\'s, volume and sig for feats with r,c in indexList'
        """pixOut is for peak dimensions, comPixOut is for centre of mass
        routine, pixWidth is for peak integration (how many Da each pixel 
        corresponds to, if we care)."""

        self.Syx = self.syx()  # load this into RAM so it runs quicker!

        featList = np.zeros((len(indexList), 4))

        def process_peak(indices):
            peak = self.getPeak(indices[0], indices[1], pixOut=pixOut)
            com_r, com_c = peak.com(pixEachSide=comPixOut)

            if cutPeaks:
                if useT4R:
                    template = peak.template4ray(integThresh=integThresh)
                else:
                    template = peak.templateNray(integThresh=integThresh, \
                                                numRays=numRays, perimOffset=perimOffset)
                peak.cutPeak(template)
            else:
                template = None

            peakVol = peak.bivSplIntegrate(pixWidth=pixWidth)
            if sampling == 'jackknife':
                peakVar = peak.jkResampleVar(cutPeak=cutPeaks, template=template)
            elif sampling == 'bootstrap':
                peakVar = peak.bsResampleVar(bsRS, cutPeak=cutPeaks, \
                                            template=template)
            elif sampling == 'bootstrap_with_diffs':
                peakVar = peak.bsDiffResampleVar(bsRS_diffs, cutPeak=cutPeaks, \
                                                template=template)
            else:
                raise TypeError('\'' + sampling + '\'' + ' not a recognised method of resampling')

            feat = round(self.scan.index2mz(indices[0] - \
                                            comPixOut + com_r), 2), round(self.scan.index2mz(indices[1] - \
                                                                                            comPixOut + com_c),
                                                                        2), peakVol, peakVol / np.sqrt(peakVar)

            return feat

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_peak, indices) for indices in indexList]

            featNo = 0
            for future in concurrent.futures.as_completed(futures):
                featList[featNo] = future.result()
                featNo += 1
                if featNo % printAt == 0:
                    print('sig calculated for feature ' + str(featNo))
                if type(saveAt) is int and featNo % saveAt == 0:
                    np.save(basePath + '_feats' + str(int(featNo - saveAt + 1)) + 'to' + \
                            str(int(featNo)) + '.npy', featList[featNo - saveAt:featNo])

        return featList
    
    
    def sampleFeatsIndex(self, indexList, pixOut=15, comPixOut=3,
                    cutPeaks=True, integThresh=0, numRays=100, perimOffset=-0.5,
                    pixWidth=1, sampling='jackknife', bsRS=None, bsRS_diffs=None,
                    useT4R=False, saveAt=False, basePath=None, printAt=50):
        'Returns m/z\'s, volume and sig for feats with r,c in indexList'
        """pixOut is for peak dimensions, comPixOut is for centre of mass
        routine, pixWidth is for peak integration (how many Da each pixel 
        corresponds to, if we care)."""

        self.Syx = self.syx()  # load this into RAM so it runs quicker!

        featList = np.zeros((len(indexList), 4))

        def process_peak(indices):
            peak = self.getPeak(indices[0], indices[1], pixOut=pixOut)
            com_r, com_c = peak.com(pixEachSide=comPixOut)

            if cutPeaks:
                if useT4R:
                    template = peak.template4ray(integThresh=integThresh)
                else:
                    template = peak.templateNray(integThresh=integThresh, \
                                                numRays=numRays, perimOffset=perimOffset)
                peak.cutPeak(template)
            else:
                template = None

            peakVol = peak.bivSplIntegrate(pixWidth=pixWidth)
            if sampling == 'jackknife':
                peakVar = peak.jkResampleVar(cutPeak=cutPeaks, template=template)
            elif sampling == 'bootstrap':
                peakVar = peak.bsResampleVar(bsRS, cutPeak=cutPeaks, \
                                            template=template)
            elif sampling == 'bootstrap_with_diffs':
                peakVar = peak.bsDiffResampleVar(bsRS_diffs, cutPeak=cutPeaks, \
                                                template=template)
            else:
                raise TypeError('\'' + sampling + '\'' + ' not a recognised method of resampling')

            feat = indices[0], indices[1], peakVol, peakVol / np.sqrt(peakVar)

            return feat

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_peak, indices) for indices in indexList]

            featNo = 0
            for future in concurrent.futures.as_completed(futures):
                featList[featNo] = future.result()
                featNo += 1
                if featNo % printAt == 0:
                    print('sig calculated for feature ' + str(featNo))
                if type(saveAt) is int and featNo % saveAt == 0:
                    np.save(basePath + '_feats' + str(int(featNo - saveAt + 1)) + 'to' + \
                            str(int(featNo)) + '.npy', featList[featNo - saveAt:featNo])

        return featList
    
    def topNfeats(self, numFeats, clearRad=25, chemFilt=[], chemFiltTol=2.0,
                  shapeFilt=False, shFiltThr=-0.2, shFiltRef='map',\
                  shFiltRad=15, boundFilt=True, boundLimit=15, breakTime=3600,\
                  returnDiscards=False):
        #chemFiltTol in Da, chemFilt condition is less than or equal to.
        #shFiltThr is fraction of highest peak on a/c cut pCov map.
        """This looks like it is written by a moron - partially because it
        was, and partially because I chose not to optimise it much because I
        currently have more pressing things to do.
        'returnDiscards' allows to return features discarded by any applied
        filters as well (this array is the second element in the tuple)."""

        array=np.triu(cutAC(self.array))
        if shFiltRef=='map': #'shape filter reference' taken globally from map
            shFiltThrAbs=shFiltThr*np.nanmax(array)

        #'Picks the top N highest legitimate features'
        feats=np.zeros((numFeats, 2))
        featCount=0

        if returnDiscards:
            discardFeats=[] #list of all feats discarded by any of the filters
            #not declared as array because you do not know how many features
            #will be discarded (it could certainly be more than those which
            #aren't discarded)

        circListClear=circList(clearRad)
        if shapeFilt:
            circListFilt=circList(shFiltRad)

        startTime=time.time()

        while featCount<numFeats:

            featPass=True
            r,c = maxIndices(array)
            if shFiltRef=='peak': #shape filter reference taken as height of
                shFiltThrAbs=shFiltThr*array[r,c] #each individual peak

            #Apply chemical filter if requested
            for chemFiltmz in chemFilt:
                if abs(self.scan.index2mz(r) - chemFiltmz) <= chemFiltTol or \
                abs(self.scan.index2mz(c) - chemFiltmz)  <= chemFiltTol: #this
                #takes m/z of highest pixel, not m/z of CoM of feature
                    featPass=False
                    break

            #Apply shape filter if requested
            if shapeFilt and featPass:
                for x in circListFilt:
                    if array[r+x[0],c+x[1]]<=shFiltThrAbs or \
                    array[r+x[0],c-x[1]]<=shFiltThrAbs or \
                    array[r-x[0],c+x[1]]<=shFiltThrAbs or \
                    array[r-x[0],c-x[1]]<=shFiltThrAbs:
                        featPass=False
                        break

            #Apply boundary filter so that features too close to the edge of
            #the map to be sampled are not counted
            if boundFilt and featPass:
                if r < boundLimit or c < boundLimit or r  > len(array) - \
                (boundLimit+1) or c > len(array) - (boundLimit+1):
                    featPass=False

            #Because of the way Python was compiling (everything
            #indented relative to a for clause, including the continue
            #statement, is not executed as soon as the break statement is hit,
            #even if it is unindented relative to the break statement),
            #there was no possible configuration of indentations for the
            #boundFilt and shapeFilt to break out of the for loops doing the
            #testing and then continue to the top of the main while loop, so I
            #used the flag variable featPass instead

            if featPass:
                feats[featCount,0], feats[featCount,1] = r, c
                featCount+=1

                if featCount%100==0:
                    print('found '+str(featCount)+' good features')

            elif returnDiscards:
                discardFeats.append([r, c])

            clearCirc(array, r, c, circListClear)

            if time.time()-startTime>breakTime:
                print('topNfeats breaking out at '+str(featCount)+' features'\
                +' - running time exceeded '+str(breakTime)+' secs')
                if not returnDiscards:
                    return feats[:featCount] #cut to the last appended feature
                else:
                    return feats[:featCount], np.array(discardFeats)
                    #discardFeats was a list so is converted to array for
                    #consistency of output

        if not returnDiscards:
            return feats
        else:
            return feats, np.array(discardFeats) #discardFeats was a list
            #so is converted to array for consistency of output




class CovMap(Map):
    'Simple covariance map belonging to a scan'

    def build(self):
        'Constructs covariance map according to covariance equation'
        syx = self.syx()

        sx = np.matrix(self.scan.scanList[:self.numScans].sum(axis=0))
        sysx = np.matrix.transpose(sx) * sx

        self.array = np.array((syx - (sysx / self.numScans)) / (self.numScans - 1))

    def getPeak(self, r, c, pixOut=15):
        return CovPeak(self, r, c, pixOut=pixOut)


class PCovMap(Map):

    def __init__(self, scan, pCovParams, numScans='all'):
        if numScans == 'all':
            self.pCovParams = pCovParams
        else:
            self.pCovParams = pCovParams[:numScans]
        Map.__init__(self, scan, numScans)

    def build(self):
        """Calculate full partial covariance map with single partial
            covariance parameter pCovParams"""

        mapScanList = self.scan.scanList[:self.numScans, :]
        var = varII(self.pCovParams)
        covXIvec = np.matrix(covXI(mapScanList, self.pCovParams))  # made matrix \
        # type (row vector) for subsequent matrix multiplication

        if os.path.isfile(self.scan.scanFolder + '/avYX.npy'):
            avYX = np.load(self.scan.scanFolder + '/avYX.npy')
        else:
            avYX = self.syx() / (self.numScans - 1)
            np.save(self.scan.scanFolder + '/avYX.npy', avYX)

        if os.path.isfile(self.scan.scanFolder + '/avYavX.npy'):
            avYavX = np.load(self.scan.scanFolder + '/avYavX.npy')
        else:
            SxVec = np.matrix(mapScanList.sum(axis=0))  # also made matrix type
            # (row vector) for subsequent matrix multiplication
            avYavX = np.array(np.matrix.transpose(SxVec) * SxVec) / \
                     (self.numScans * (self.numScans - 1))
            np.save(self.scan.scanFolder + '/avYavX.npy', avYavX)

        self.array = (self.numScans - 1) / (self.numScans - 2) * (avYX - avYavX - \
                                                                  np.array(
                                                                      np.matrix.transpose(covXIvec) * covXIvec) / var)

    def getPeak(self, r, c, pixOut=15):
        return PCovPeak(self, r, c, pixOut=pixOut)

    def Si(self):
        return self.pCovParams.sum(axis=0)

    def Si2(self):
        return (self.pCovParams ** 2).sum(axis=0)


class Peak:
    'Any peak from a 2D map'

    def __init__(self, array):
        self.array = array

    def com(self, pixEachSide=3):
        """Returns the row and column index of the centre of mass of a square 
        on the 2D array 'array', centred on the pixel indexed by r, c and of 
        width (2 * pixEachSide + 1)"""

        square = np.zeros((2 * pixEachSide + 1, 2 * pixEachSide + 1))

        for i in range(2 * pixEachSide + 1):
            for j in range(2 * pixEachSide + 1):
                square[i, j] = self.array[self.pixOut - pixEachSide + i, \
                                          self.pixOut - pixEachSide + j]
                # It is clearer to cast the indexing of the array like this
                # because it is consistent with how the index of the COM is
                # returned

        # Having a negative value in 'square' can cause the centre of mass formula
        # to return a value outside of the boundaries of 'square'. This is
        # undesirable for the purposes of this function (a result of the way in
        # which negative values are interpreted on a CV/pCV map).
        # So we check if there are negative values in 'square', and if so we get
        # rid of them by uniformly raising the values of 'square' so that the
        # minimum value is not negative but zero (and all other values are
        # positive). This allows the centre-of-mass formula to provide the required
        # indices for our purpose here.
        squareMin = np.nanmin(square)
        if squareMin < 0:
            squareTwo = abs(squareMin) * \
                        np.ones((2 * pixEachSide + 1, 2 * pixEachSide + 1))
            square += squareTwo

        COMi, COMj = ndimage.measurements.center_of_mass(square)

        return COMi, COMj  # self.r-pixEachSide+COMi, self.c-\
        # pixEachSide+COMj

    def template4ray(self, integThresh=0):
        "This should be replaced with the improved templateNray() method"
        """peakSquare must be square"""

        north = south = east = west = 1
        valueN = valueS = valueE = valueW = integThresh + 1  # to make sure it
        # starts off above the integThresh

        peakShape = np.ones((len(self.array), \
                             len(self.array)), dtype=bool)

        rad, rem = divmod(len(self.array), 2)
        if rem != 1:
            raise ValueError('square width not odd integer -> ambiguous apex')

        while valueN > integThresh:
            north += 1
            if north <= rad:
                valueN = self.array[rad + north, rad]
            else:
                break
        north -= 1  # gives index of last pixel above threshold

        while valueS > integThresh:
            south += 1
            if south <= rad:
                valueS = self.array[rad - south, rad]
            else:
                break
        south -= 1  # gives index of last pixel above threshold

        while valueE > integThresh:
            east += 1
            if east <= rad:
                valueE = self.array[rad, rad + east]
            else:
                break
        east -= 1  # gives index of last pixel above threshold

        while valueW > integThresh:
            west += 1
            if west <= rad:
                valueW = self.array[rad, rad - west]
            else:
                break
        west -= 1  # gives index of last pixel above threshold

        # Now cut out the parts of the square you don't want
        for plusi in range(north + 1, rad + 1):
            peakShape[rad + plusi, rad] = False

        for minusi in range(south + 1, rad + 1):
            peakShape[rad - minusi, rad] = False

        for plusj in range(east + 1, rad + 1):
            peakShape[rad, rad + plusj] = False

        for minusj in range(west + 1, rad + 1):
            peakShape[rad, rad - minusj] = False

        # Quadrant 1
        m = -north / float(east)
        for j1 in range(1, rad + 1):
            boundUp = m * j1 + north
            for i1 in range(1, rad + 1):
                if i1 > boundUp:
                    peakShape[rad + i1, rad + j1] = False

        # Quadrant 2
        m = south / float(east)
        for j2 in range(1, rad + 1):
            boundDown = m * j2 - south
            for i2 in range(-1, -(rad + 1), -1):
                if i2 < boundDown:
                    peakShape[rad + i2, rad + j2] = False

        # Quadrant 3
        m = -south / float(west)
        for j3 in range(-1, -(rad + 1), -1):
            boundDown1 = m * j3 - south
            for i3 in range(-1, -(rad + 1), -1):
                if i3 < boundDown1:
                    peakShape[rad + i3, rad + j3] = False
        #            
        # Quadrant 4
        m = north / float(west)
        for j4 in range(-1, -(rad + 1), -1):
            boundUp1 = m * j4 + north
            for i4 in range(1, rad + 1):
                if i4 > boundUp1:
                    peakShape[rad + i4, rad + j4] = False

        return peakShape

    def templateNray(self, numRays, perimOffset=-0.5, integThresh=0, \
                     maxRayLength='square_width', r_i='square_centre', c_i='square_centre'):
        "Return boolean template of peak, created by joining end of N rays"
        """integThresh condition is <=. r_i and c_i are row and column indices
        to cast rays from. perimOffset is perimeter offset - offset between
        vertices and the perimeter of the template outline (passed as radius 
        to Path.contains_points() method)."""

        array = self.array
        dim = len(array)  # if dim odd, maxRayLength falls one pixel short of
        # 'bottom' and 'right' edges of array
        cent = int(np.floor((dim - 1) / 2))  # either central pixel if dim is odd or
        # 'top left' of central 4 pixels if dim is even

        if maxRayLength == 'square_width':
            maxRayLength = cent
        if r_i == 'square_centre':
            r_i = cent
        if c_i == 'square_centre':
            c_i = cent

        vertices = []  # first point at end of each ray where value<=integThresh
        for theta in np.linspace(0, 2 * np.pi, numRays, endpoint=False):
            # endpoint=False because ray at 2*pi is same direction as ray at 0
            r = r_i
            c = c_i

            incR = np.cos(theta)  # increment in Row index - so first ray cast
            # directly 'down' (when theta==0, cos(theta)=1, sin(theta)=0)
            incC = np.sin(theta)  # increment in Column index

            for x in range(maxRayLength):
                r += incR
                c += incC
                if array[int(np.round(r)), int(np.round(c))] <= integThresh:
                    if (np.round(r), np.round(c)) not in vertices:
                        vertices.append((np.round(r), np.round(c)))
                    break
            else:  # this is equivalent to saying the pixel the next step out
                # would have been below the integThresh
                r += incR
                c += incC
                vertices.append((np.round(r), np.round(c)))

        vertices = Path(vertices)  # instance of matplotlib.path.Path class,
        # efficiently finds points within arbitrary polygon defined by
        # vertices

        points = np.zeros((dim ** 2, 2), dtype=int)
        points[:, 0] = np.repeat(np.arange(0, dim, 1), dim)
        points[:, 1] = np.tile(np.arange(0, dim, 1), dim)

        points = points[vertices.contains_points(points, radius=perimOffset)]
        # only choose those points which are inside the polygon traced by the
        # cast rays. 'radius' kwarg poorly documented but setting to -0.5 draws
        # polygon inside vertices as required (because vertices are elements
        # with value <= thresh).
        template = np.zeros((dim, dim), dtype=bool)
        for x in points:
            template[x[0], x[1]] = True

        return template

    def cutPeak(self, peakTemplate):
        self.array[peakTemplate == False] = 0

    def bivSplIntegrate(self, pixWidth=1):  # could also have
        # pixWidth=self.fromMap.scan.sliceSize
        mesh_array = np.arange(len(self.array)) * pixWidth  # bivariate spline
        # requires a meshgrid. Because the MS gives out arbitrary units in the
        # text files used to make the CV maps (and this function has been the
        # only one so far used to extract anything quantitative from the maps),
        # the spacing of the elements (=pixWidth) in the two vectors to define
        # this meshgrid has been unimportant (provided it is uniform) and for
        # convenience has by default been unity. To maintain consistency for
        # testing, pixWidth is therefore currently set to 1 when this function
        # is called from the sampleFeats. If required, pixWidth can (and should,
        # really) of course be set to sliceSize.
        spline = interpolate.RectBivariateSpline(mesh_array, mesh_array, \
                                                 self.array)
        # make the bivariate spline, default degree is 3
        return spline.integral(np.nanmin(mesh_array), np.nanmax(mesh_array), \
                               np.nanmin(mesh_array), np.nanmax(mesh_array))
        # returns the integral across this spline

    def lineout(self, axis, out=0):
        # the 'x' lineout is the lineout *across* the c dimension, and so the
        # lineout *along* the m/z indexed by self.r.
        if axis == 'x':
            line = self.array[self.pixOut - out:self.pixOut + out + 1, \
                   :].sum(0)
        elif axis == 'y':
            line = self.array[:, self.pixOut - out: \
                                 self.pixOut + out + 1].sum(1)
        return line

    def fwhm(self, axis, plot=False, inDa=False, outforlo=0):
        # the 'x' FWHM is the FWHM *across* the c dimension, and so the FWHM 
        # *along* the m/z indexed by self.r. outforlo is 'pixels out for line\
        # out', i.e. take the lineout for the univariate spline going out how
        # many pixels each side?
        line = self.lineout(axis, out=outforlo)

        xs = np.arange(self.pixOut * 2 + 1) * (self.fromMap.scan.sliceSize if \
                                                   inDa else 1.)
        spline = interpolate.UnivariateSpline(xs, line - \
                                              line[self.pixOut] / 2., s=0)
        # s=0 -> spline interpolates through all data points
        roots = spline.roots()
        if plot:
            fig0 = plt.figure()
            ax0 = fig0.add_subplot(111)
            ax0.plot(xs, line)
            xsfine = np.arange(self.pixOut * 2 + 1, step=0.1) * \
                     (self.fromMap.scan.sliceSize if inDa else 1.)
            ax0.plot(xsfine, spline(xsfine) + line[self.pixOut] / 2.)  # add half
            # the max back on for the plotting
            ax0.vlines(roots, np.nanmin(line), np.nanmax(line))
            plt.show()

        if len(roots) != 2:  # return nan if more (or fewer)
            # than 2 roots are found
            return np.nan
        else:
            return abs(roots[1] - roots[0])


class tempPeak(Peak):
    pass  # this is so that when you resample you can e.g. integrate and find
    # the CoM if for some reason you would like to.


class CovPeak(Peak):

    def __init__(self, fromMap, r, c, pixOut=15):
        self.fromMap = fromMap
        self.r = int(r)
        self.c = int(c)
        self.pixOut = int(pixOut)
        self.build()

    def build(self):
        numScans = self.fromMap.numScans
        self.array = (self.Syx() - self.SySx() / (numScans)) / (numScans - 1)

    def jkResampleVar(self, cutPeak=True, template=None):

        numScans = self.fromMap.numScans
        yScans = self.yScans()
        xScans = self.xScans()
        SyxFull = self.Syx()
        SyFull = self.Sy()
        SxFull = self.Sx()

        covSum = 0
        covSumSqd = 0

        for missingScan in range(numScans):

            Syx = SyxFull - \
                  np.array(np.matrix.transpose(np.matrix(yScans[missingScan, :])) \
                           * np.matrix(xScans[missingScan, :]))

            Sy = SyFull - yScans[missingScan, :]
            Sx = SxFull - xScans[missingScan, :]

            SySx = np.matrix.transpose(np.matrix(Sy)) * \
                   np.matrix(Sx)

            # Number of scans for each partial covariance square on the resample
            # = numScans - 1            
            covSquare = Peak((Syx - SySx / (numScans - 1)) / (numScans - 2))

            if cutPeak:
                covSquare.cutPeak(template)

            vol = covSquare.bivSplIntegrate()

            covSum += vol
            covSumSqd += vol ** 2

        return (covSumSqd - (covSum ** 2) / (numScans)) / numScans * (numScans-1)  # this should
        # include Bessel's correction, but has not historically. Because we are
        # currently unconcerned with absolute value of stdDev but do want
        # to compare with previous results, omit for now.

    def reCentre(self, maxChange=3, printOut=False):
        """For when you are not entirely sure where the maximum is. Max change
        is the number of pixels you are willing to change x and y by."""
        rf, cf = maxIndices(self.array \
                                [self.pixOut - maxChange:self.pixOut + maxChange + 1, \
                            self.pixOut - maxChange:self.pixOut + maxChange + 1])

        if rf == maxChange and cf == maxChange and printOut:
            print('no shift in peak apex')
        elif cf == maxChange:
            if printOut:
                print('r shifted down by ' + str(rf - maxChange) + ' pixels')
            self.r += rf - maxChange
            self.build()
        elif rf == maxChange:
            if printOut:
                print('c shifted right by ' + str(cf - maxChange) + ' pixels')
            self.c += cf - maxChange
            self.build()
        else:
            if printOut:
                print('r shifted down by ' + str(rf - maxChange) + \
                      ' pixels and c shifted right by ' + str(cf - maxChange) + ' pixels')
            self.r += rf - maxChange
            self.c += cf - maxChange
            self.build()

    def reCentre2(self, maxChange=3):
        """For when you are not entirely sure where the maximum is. Max change
        is the number of pixels you are willing to change x and y by. Instead
        of printing, this returns change_in_r,change_in_c and rebuilds the 
        peak array (as well as changing self.r & self.c)"""
        rf, cf = maxIndices(self.array \
                                [self.pixOut - maxChange:self.pixOut + maxChange + 1, \
                            self.pixOut - maxChange:self.pixOut + maxChange + 1])

        if rf == maxChange and cf == maxChange:  # no need to rebuild in this
            # case
            return 0, 0  # no change in r or c
        else:
            self.r += rf - maxChange
            self.c += cf - maxChange
            self.build()
            return rf - maxChange, cf - maxChange

    def Sy(self):
        return self.yScans().sum(axis=0)

    def Sx(self):
        return self.xScans().sum(axis=0)

    def Syx(self):
        fromIndexRow = self.r - self.pixOut
        toIndexRow = self.r + self.pixOut
        fromIndexCol = self.c - self.pixOut
        toIndexCol = self.c + self.pixOut
        return self.fromMap.syx()[fromIndexRow:toIndexRow + 1, \
               fromIndexCol:toIndexCol + 1]

    def SySx(self):
        return np.array(np.matrix.transpose(np.matrix(self.Sy())) * \
                        np.matrix(self.Sx()))

    def yScans(self):
        fromIndexRow = self.r - self.pixOut
        toIndexRow = self.r + self.pixOut
        return self.fromMap.scan.scanList[:self.fromMap.numScans, \
               fromIndexRow:toIndexRow + 1]

    def xScans(self):
        fromIndexCol = self.c - self.pixOut
        toIndexCol = self.c + self.pixOut
        return self.fromMap.scan.scanList[:self.fromMap.numScans, \
               fromIndexCol:toIndexCol + 1]


class PCovPeak(CovPeak):

    def build(self):
        numScans = self.fromMap.numScans

        Si2 = self.fromMap.Si2()
        Si = self.fromMap.Si()
        S2i = Si ** 2

        varII = (Si2 - S2i / numScans) / \
                (numScans - 1)

        Syx = self.Syx()
        SySx = self.SySx()

        SiSx = self.SiSx()
        SySi = self.SySi()

        Six = self.Six()
        Syi = self.Syi()

        covYX = (Syx - SySx / (numScans)) / (numScans - 1)
        covYI = (Syi - SySi / (numScans)) / (numScans - 1)
        covIX = (Six - SiSx / (numScans)) / (numScans - 1)

        self.array = ((numScans - 1) / (numScans - 2)) * (covYX - \
                                                          np.array(np.matrix.transpose(np.matrix(covYI)) \
                                                                   * np.matrix(covIX)) / varII)

    def bsResampleVar(self, resamps, cutPeak=True, template=None):
        #        time1=time.time()
        numScans = self.fromMap.numScans
        pCovParams = self.fromMap.pCovParams
        yScans = self.yScans()
        xScans = self.xScans()

        numRS = len(resamps)

        pCovSum = 0
        pCovSumSqd = 0

        for rs in resamps:
            yScansRS = yScans[rs]
            xScansRS = xScans[rs]
            pCovParsRS = pCovParams[rs]

            syx = np.einsum('ij,ik->jk', yScansRS, xScansRS)
            sy = yScansRS.sum(axis=0)
            sx = xScansRS.sum(axis=0)
            sysx = np.outer(sy, sx)  # this was found to be quickest o.p. routine on
            # http://stackoverflow.com/questions/27809511/efficient-outer-product-in-python
            si = pCovParsRS.sum()
            si2 = (np.power(pCovParsRS, 2)).sum()

            sisx = si * sx
            sysi = sy * si
            six = np.einsum('ij,i->j', xScansRS, pCovParsRS)
            syi = np.einsum('ij,i->j', yScansRS, pCovParsRS)

            covYX = (syx - sysx / (numScans)) / (numScans - 1)  # the bootstrap resample has
            covYI = (syi - sysi / (numScans)) / (numScans - 1)  # self.numScans in it, whilst
            covIX = (six - sisx / (numScans)) / (numScans - 1)  # the jackknife resample only
            varII = (si2 - np.power(si, 2) / (numScans)) / (numScans - 1)  # has self.numScans-1
            # scans in it
            pCovSquare = Peak(((numScans - 1) / (numScans - 2)) * \
                              (covYX - np.array(np.matrix.transpose(np.matrix(covYI)) * \
                                                np.matrix(covIX)) / varII))  # factors of numScans-1 could cancel out

            if cutPeak:
                pCovSquare.cutPeak(template)

            vol = pCovSquare.bivSplIntegrate()

            pCovSum += vol
            pCovSumSqd += vol ** 2

        #        print 'finished BS resampling one peak, time taken = '+\
        #        str(time.time()-time1)+' s'
        return (pCovSumSqd - (pCovSum ** 2) / (numRS)) / numRS  # this should include
        # Bessel's correction, but has not historically. Because we are
        # currently unconcerned with absolute value of stdDev but do want
        # to compare with previous results, omit for now.

    def bsDiffResampleVar(self, diffs, cutPeak=True, template=None):
        """"Difference from original scan set only. Row 0 of diffs is all zeroes-
        original scan set"""
        #        time1=time.time()
        numScans = self.fromMap.numScans
        pCovParams = self.fromMap.pCovParams
        yScans = self.yScans()
        xScans = self.xScans()

        syxFull = self.Syx()
        syFull = self.Sy()
        sxFull = self.Sx()
        siFull = self.fromMap.Si()
        si2Full = self.fromMap.Si2()
        sixFull = self.Six()
        syiFull = self.Syi()

        numRS = len(diffs) - 1  # row 0 of diffs is all zeroes - original scan set

        pCovSum = 0
        pCovSumSqd = 0

        for x in range(1, len(diffs)):  # row 0 of diffs is all zeroes -
            # original scan set

            diff = diffs[x]
            mask = diff < 0  # if element of rs is less than 0, will only ever be -1 if
            # comparing to the original scan set
            yScansTake = yScans[mask]
            xScansTake = xScans[mask]
            pCovParsTake = pCovParams[mask]

            syTake = yScansTake.sum(axis=0)
            sxTake = xScansTake.sum(axis=0)
            siTake = pCovParsTake.sum()

            sixTake = np.einsum('ij,i->j', xScansTake, pCovParsTake)
            syiTake = np.einsum('ij,i->j', yScansTake, pCovParsTake)
            syxTake = np.einsum('ij,ik->jk', yScansTake, xScansTake)
            si2Take = (np.power(pCovParsTake, 2)).sum()

            # Now what to add
            rep = np.copy(diff)
            rep[mask] = 0

            yScansAdd = np.repeat(yScans, rep, axis=0)
            xScansAdd = np.repeat(xScans, rep, axis=0)
            pCovParsAdd = np.repeat(pCovParams, rep, axis=0)

            syAdd = yScansAdd.sum(axis=0)
            sxAdd = xScansAdd.sum(axis=0)
            siAdd = pCovParsAdd.sum()

            sixAdd = np.einsum('ij,i->j', xScansAdd, pCovParsAdd)
            syiAdd = np.einsum('ij,i->j', yScansAdd, pCovParsAdd)
            syxAdd = np.einsum('ij,ik->jk', yScansAdd, xScansAdd)
            si2Add = (np.power(pCovParsAdd, 2)).sum()

            # Now calculate adjusted measures
            syx = syxFull - syxTake + syxAdd
            six = sixFull - sixTake + sixAdd
            syi = syiFull - syiTake + syiAdd

            # Now those measures that are used more than once in below calcula-
            # tions
            sy = syFull - syTake + syAdd
            sx = sxFull - sxTake + sxAdd
            si = siFull - siTake + siAdd
            si2 = si2Full - si2Take + si2Add

            sysx = np.outer(sy, sx)  # this was found to be quickest o.p. routine on
            # http://stackoverflow.com/questions/27809511/efficient-outer-product-in-python
            sisx = si * sx
            sysi = sy * si

            covYX = (syx - sysx / (numScans)) / (numScans - 1)  # the bootstrap resample has
            covYI = (syi - sysi / (numScans)) / (numScans - 1)  # self.numScans in it, whilst
            covIX = (six - sisx / (numScans)) / (numScans - 1)  # the jackknife resample only
            varII = (si2 - np.power(si, 2) / (numScans)) / (numScans - 1)  # has self.numScans-1
            # scans in it
            pCovSquare = Peak(((numScans - 1) / (numScans - 2)) * \
                              (covYX - np.array(np.matrix.transpose(np.matrix(covYI)) * \
                                                np.matrix(covIX)) / varII))  # factors of numScans-1 could cancel out

            if cutPeak:
                pCovSquare.cutPeak(template)

            vol = pCovSquare.bivSplIntegrate()

            pCovSum += vol
            pCovSumSqd += vol ** 2

        #        print 'finished BS resampling w/diffs for one peak, time taken = '+\
        #        str(time.time()-time1)+' s'
        return (pCovSumSqd - (pCovSum ** 2) / (numRS)) / numRS  # this is
        # without Bessel's correction

    def jkResampleVar(self, cutPeak=True, template=None):

        numScans = self.fromMap.numScans
        pCovParams = self.fromMap.pCovParams
        yScans = self.yScans()
        xScans = self.xScans()
        SyxFull = self.Syx()
        SyFull = self.Sy()
        SxFull = self.Sx()
        SiFull = self.fromMap.Si()
        Si2Full = self.fromMap.Si2()
        SixFull = self.Six()
        SyiFull = self.Syi()

        pCovSum = 0
        pCovSumSqd = 0

        for missingScan in range(numScans):

            Syx = SyxFull - \
                  np.array(np.matrix.transpose(np.matrix(yScans[missingScan, :])) \
                           * np.matrix(xScans[missingScan, :]))

            Sy = SyFull - yScans[missingScan, :]
            Sx = SxFull - xScans[missingScan, :]

            SySx = np.matrix.transpose(np.matrix(Sy)) * \
                   np.matrix(Sx)

            Si = SiFull - pCovParams[missingScan]
            Si2 = Si2Full - (pCovParams[missingScan]) ** 2

            SiSx = Si * Sx
            SySi = Sy * Si

            Six = SixFull - pCovParams[missingScan] * xScans[missingScan, :]
            Syi = SyiFull - yScans[missingScan, :] * pCovParams[missingScan]

            # Number of scans for each partial covariance square on the resample
            # = numScans - 1

            covYX = (Syx - SySx / (numScans - 1)) / (numScans - 2)
            covYI = (Syi - SySi / (numScans - 1)) / (numScans - 2)
            covIX = (Six - SiSx / (numScans - 1)) / (numScans - 2)
            varII = (Si2 - Si ** 2 / (numScans - 1)) / (numScans - 2)

            pCovSquare = Peak(((numScans - 2) / (numScans - 3)) * \
                              (covYX - np.array(np.matrix.transpose(np.matrix(covYI)) * \
                                                np.matrix(covIX)) / varII))  # factors of numScans-2 could cancel out

            if cutPeak:
                pCovSquare.cutPeak(template)

            vol = pCovSquare.bivSplIntegrate()

            pCovSum += vol
            pCovSumSqd += vol ** 2

        return (pCovSumSqd - (pCovSum ** 2) / (numScans)) / numScans * (numScans-1)  # this should
        # include Bessel's correction, but has not historically. Because we are
        # currently unconcerned with absolute value of stdDev but do want
        # to compare with previous results, omit for now.

    def SiSx(self):
        return self.fromMap.Si() * self.Sx()

    def SySi(self):
        return self.Sy() * self.fromMap.Si()

    def Syi(self):
        SyiPeak = np.zeros(self.pixOut * 2 + 1)
        yScans = self.yScans()
        for scanIndex in range(self.fromMap.numScans):
            SyiPeak += yScans[scanIndex, :] * self.fromMap.pCovParams[scanIndex]
        return SyiPeak

    def Six(self):
        SixPeak = np.zeros(self.pixOut * 2 + 1)
        xScans = self.xScans()
        for scanIndex in range(self.fromMap.numScans):
            SixPeak += xScans[scanIndex, :] * self.fromMap.pCovParams[scanIndex]
        return SixPeak

# %%

# %%

# %%
