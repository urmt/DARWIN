# src/sfh/core/coherence_model.py
"""
Coherence Model Implementation for Sentience-Field Hypothesis
Based on Chapter 4: Mathematical Framework
"""

import numpy as np
from scipy import optimize
from typing import Tuple, Optional, Dict, Any

class CoherenceModel:
    """
    Implements the coherence component of SFH optimization.
    
    Coherence (C) represents structural stability and information integration
    within biological systems.
    """
    
    def __init__(self, 
                 alpha: float = 1.0, 
                 beta: float = 0.5,
                 gamma: float = 0.1):
        """
        Initialize coherence model parameters.
        
        Args:
            alpha: Base coherence scaling factor
            beta: Network connectivity weight
            gamma: Stability damping coefficient
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def calculate_coherence(self, 
                          connectivity_matrix: np.ndarray,
                          stability_vector: np.ndarray,
                          time_step: float = 1.0) -> float:
        """
        Calculate system coherence based on connectivity and stability.
        
        C(t) = α * (Σᵢⱼ Mᵢⱼ * exp(-γt)) + β * Σᵢ Sᵢ(t)
        
        Args:
            connectivity_matrix: N×N matrix of system connections
            stability_vector: N-dimensional stability measures
            time_step: Current time step
            
        Returns:
            Coherence value
        """
        # Network connectivity contribution
        connectivity_sum = np.sum(connectivity_matrix * np.exp(-self.gamma * time_step))
        
        # Stability contribution
        stability_sum = np.sum(stability_vector)
        
        coherence = self.alpha * connectivity_sum + self.beta * stability_sum
        
        return max(0, coherence)  # Coherence cannot be negative
    
    def coherence_gradient(self,
                          connectivity_matrix: np.ndarray,
                          stability_vector: np.ndarray,
                          perturbation: float = 1e-6) -> np.ndarray:
        """
        Calculate gradient of coherence with respect to system parameters.
        
        Returns:
            Gradient vector for optimization
        """
        n = len(stability_vector)
        gradient = np.zeros(n)
        
        base_coherence = self.calculate_coherence(connectivity_matrix, stability_vector)
        
        for i in range(n):
            # Perturb stability vector
            perturbed_stability = stability_vector.copy()
            perturbed_stability[i] += perturbation
            
            perturbed_coherence = self.calculate_coherence(
                connectivity_matrix, perturbed_stability
            )
            
            gradient[i] = (perturbed_coherence - base_coherence) / perturbation
        
        return gradient


# src/sfh/core/fertility_model.py
"""
Fertility Model Implementation for Sentience-Field Hypothesis
Based on Chapter 4: Mathematical Framework
"""

import numpy as np
from typing import List, Tuple

class FertilityModel:
    """
    Implements the fertility component of SFH optimization.
    
    Fertility (F) represents generative potential and adaptive capacity
    of biological systems.
    """
    
    def __init__(self,
                 delta: float = 2.0,
                 epsilon: float = 1.0,
                 zeta: float = 0.3):
        """
        Initialize fertility model parameters.
        
        Args:
            delta: Reproductive rate scaling
            epsilon: Innovation potential weight
            zeta: Resource efficiency coefficient
        """
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta
    
    def calculate_fertility(self,
                           reproductive_rate: float,
                           innovation_potential: np.ndarray,
                           resource_efficiency: float) -> float:
        """
        Calculate system fertility based on reproduction and innovation.
        
        F(t) = δ * R(t) + ε * Σᵢ I_potential,ᵢ + ζ * E_resource(t)
        
        Args:
            reproductive_rate: Current reproduction rate
            innovation_potential: Vector of innovation capabilities
            resource_efficiency: Resource utilization efficiency
            
        Returns:
            Fertility value
        """
        reproduction_term = self.delta * reproductive_rate
        innovation_term = self.epsilon * np.sum(innovation_potential)
        efficiency_term = self.zeta * resource_efficiency
        
        fertility = reproduction_term + innovation_term + efficiency_term
        
        return max(0, fertility)  # Fertility cannot be negative
    
    def fertility_dynamics(self,
                          current_fertility: float,
                          environmental_pressure: float,
                          dt: float = 0.1) -> float:
        """
        Model fertility changes over time under environmental pressure.
        
        dF/dt = F * (1 - F/K) - P_env * F
        
        Where K is carrying capacity (normalized to 1)
        """
        carrying_capacity = 1.0
        growth_rate = current_fertility * (1 - current_fertility / carrying_capacity)
        pressure_effect = environmental_pressure * current_fertility
        
        df_dt = growth_rate - pressure_effect
        new_fertility = current_fertility + df_dt * dt
        
        return max(0, new_fertility)


# src/sfh/core/optimization.py
"""
Coherence-Fertility Optimization Implementation
Based on Chapter 4: Quantitative Framework
"""

import numpy as np
from scipy import optimize
from typing import Tuple, Dict, Optional, Callable
from .coherence_model import CoherenceModel
from .fertility_model import FertilityModel

class SFHOptimizer:
    """
    Implements the core SFH optimization algorithm that balances
    coherence and fertility for evolutionary advantage.
    """
    
    def __init__(self,
                 coherence_weight: float = 0.5,
                 fertility_weight: float = 0.5,
                 constraint_penalty: float = 1000.0):
        """
        Initialize SFH optimizer.
        
        Args:
            coherence_weight: Relative importance of coherence (0-1)
            fertility_weight: Relative importance of fertility (0-1)
            constraint_penalty: Penalty for constraint violations
        """
        self.w_c = coherence_weight
        self.w_f = fertility_weight
        self.penalty = constraint_penalty
        
        # Normalize weights
        total_weight = self.w_c + self.w_f
        if total_weight > 0:
            self.w_c /= total_weight
            self.w_f /= total_weight
        
        self.coherence_model = CoherenceModel()
        self.fertility_model = FertilityModel()
    
    def objective_function(self,
                          parameters: np.ndarray,
                          system_data: Dict) -> float:
        """
        SFH objective function: maximize weighted sum of coherence and fertility.
        
        Objective = w_c * C(params) + w_f * F(params)
        
        Args:
            parameters: System parameters to optimize
            system_data: Dictionary containing system state data
            
        Returns:
            Negative objective value (for minimization)
        """
        # Extract system state
        connectivity = system_data['connectivity_matrix']
        stability = parameters[:len(parameters)//2]  # First half for stability
        innovation = parameters[len(parameters)//2:]  # Second half for innovation
        
        # Calculate coherence
        coherence = self.coherence_model.calculate_coherence(
            connectivity, stability
        )
        
        # Calculate fertility
        fertility = self.fertility_model.calculate_fertility(
            reproductive_rate=np.mean(innovation),
            innovation_potential=innovation,
            resource_efficiency=np.mean(stability)
        )
        
        # Combined objective
        objective = self.w_c * coherence + self.w_f * fertility
        
        # Add constraint penalties if needed
        penalty = 0
        if np.any(parameters < 0):  # Non-negativity constraint
            penalty += self.penalty * np.sum(np.abs(parameters[parameters < 0]))
        
        return -(objective - penalty)  # Negative for minimization
    
    def optimize_system(self,
                       initial_parameters: np.ndarray,
                       system_data: Dict,
                       method: str = 'BFGS') -> Dict:
        """
        Optimize system parameters for maximum evolutionary fitness.
        
        Args:
            initial_parameters: Starting parameter values
            system_data: System state information
            method: Optimization algorithm
            
        Returns:
            Optimization results dictionary
        """
        # Set bounds (all parameters must be non-negative)
        bounds = [(0, None) for _ in range(len(initial_parameters))]
        
        # Perform optimization
        result = optimize.minimize(
            fun=self.objective_function,
            x0=initial_parameters,
            args=(system_data,),
            method=method,
            bounds=bounds
        )
        
        # Calculate final coherence and fertility
        optimal_params = result.x
        n_params = len(optimal_params)
        stability = optimal_params[:n_params//2]
        innovation = optimal_params[n_params//2:]
        
        final_coherence = self.coherence_model.calculate_coherence(
            system_data['connectivity_matrix'], stability
        )
        
        final_fertility = self.fertility_model.calculate_fertility(
            reproductive_rate=np.mean(innovation),
            innovation_potential=innovation,
            resource_efficiency=np.mean(stability)
        )
        
        return {
            'success': result.success,
            'optimal_parameters': optimal_params,
            'final_coherence': final_coherence,
            'final_fertility': final_fertility,
            'combined_fitness': self.w_c * final_coherence + self.w_f * final_fertility,
            'iterations': result.nit,
            'function_evaluations': result.nfev,
            'optimization_message': result.message
        }
    
    def evolutionary_trajectory(self,
                               initial_state: Dict,
                               n_generations: int = 100,
                               mutation_rate: float = 0.01) -> Dict:
        """
        Simulate evolutionary trajectory using SFH optimization.
        
        Args:
            initial_state: Initial system state
            n_generations: Number of generations to simulate
            mutation_rate: Random mutation probability
            
        Returns:
            Trajectory data over generations
        """
        coherence_history = []
        fertility_history = []
        fitness_history = []
        parameter_history = []
        
        current_params = initial_state['parameters'].copy()
        system_data = initial_state['system_data']
        
        for generation in range(n_generations):
            # Add mutations
            if mutation_rate > 0:
                mutations = np.random.normal(0, mutation_rate, len(current_params))
                current_params = np.maximum(0, current_params + mutations)
            
            # Optimize current generation
            result = self.optimize_system(current_params, system_data)
            
            if result['success']:
                current_params = result['optimal_parameters']
            
            # Record history
            coherence_history.append(result['final_coherence'])
            fertility_history.append(result['final_fertility'])
            fitness_history.append(result['combined_fitness'])
            parameter_history.append(current_params.copy())
        
        return {
            'coherence_trajectory': np.array(coherence_history),
            'fertility_trajectory': np.array(fertility_history),
            'fitness_trajectory': np.array(fitness_history),
            'parameter_trajectory': np.array(parameter_history),
            'generations': n_generations
        }
