# src/sfh/dynamics/symbiotic_evolution.py
"""
Symbiotic Evolution Dynamics Implementation
Based on Chapters 5-6: Symbiosis as Primary Evolutionary Driver
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Organism:
    """Represents an organism in symbiotic evolution simulation."""
    id: int
    coherence: float
    fertility: float
    metabolic_pathways: List[str]
    symbiotic_partners: List[int]
    generation: int = 0

class SymbioticEvolution:
    """
    Simulates evolution through symbiotic cooperation rather than pure competition.
    Implements the core SFH prediction that symbiosis drives major transitions.
    """
    
    def __init__(self, 
                 cooperation_benefit: float = 1.5,
                 competition_cost: float = 0.3,
                 symbiosis_threshold: float = 0.7):
        """
        Initialize symbiotic evolution parameters.
        
        Args:
            cooperation_benefit: Fitness multiplier for cooperative interactions
            competition_cost: Fitness penalty for competitive interactions
            symbiosis_threshold: Minimum compatibility for symbiosis
        """
        self.coop_benefit = cooperation_benefit
        self.comp_cost = competition_cost
        self.symb_threshold = symbiosis_threshold
        self.organisms = []
        self.interaction_history = []
    
    def calculate_compatibility(self, org1: Organism, org2: Organism) -> float:
        """
        Calculate metabolic compatibility between two organisms.
        High compatibility enables symbiotic relationships.
        """
        # Metabolic complementarity
        pathways1 = set(org1.metabolic_pathways)
        pathways2 = set(org2.metabolic_pathways)
        
        # Complementary pathways increase compatibility
        complementary = len(pathways1.symmetric_difference(pathways2))
        total_pathways = len(pathways1.union(pathways2))
        
        if total_pathways == 0:
            return 0.0
        
        compatibility = complementary / total_pathways
        
        # Coherence similarity also matters
        coherence_diff = abs(org1.coherence - org2.coherence)
        coherence_compatibility = np.exp(-coherence_diff)
        
        return 0.7 * compatibility + 0.3 * coherence_compatibility
    
    def form_symbiosis(self, org1: Organism, org2: Organism) -> Tuple[float, float]:
        """
        Form symbiotic relationship between compatible organisms.
        Returns fitness benefits for each organism.
        """
        compatibility = self.calculate_compatibility(org1, org2)
        
        if compatibility < self.symb_threshold:
            return 0.0, 0.0  # No symbiosis formed
        
        # Mutual benefits based on compatibility
        benefit1 = self.coop_benefit * compatibility * org2.fertility
        benefit2 = self.coop_benefit * compatibility * org1.fertility
        
        # Update symbiotic partners
        if org2.id not in org1.symbiotic_partners:
            org1.symbiotic_partners.append(org2.id)
        if org1.id not in org2.symbiotic_partners:
            org2.symbiotic_partners.append(org1.id)
        
        return benefit1, benefit2
    
    def calculate_fitness(self, organism: Organism) -> float:
        """
        Calculate organism fitness including symbiotic benefits.
        SFH prediction: symbiotic organisms have higher fitness than competitors.
        """
        base_fitness = organism.coherence * organism.fertility
        
        # Symbiotic benefits
        symbiotic_benefit = 0.0
        for partner_id in organism.symbiotic_partners:
            partner = next((org for org in self.organisms if org.id == partner_id), None)
            if partner:
                compatibility = self.calculate_compatibility(organism, partner)
                symbiotic_benefit += self.coop_benefit * compatibility
        
        # Competition costs (organisms without partners suffer)
        if len(organism.symbiotic_partners) == 0:
            competition_penalty = self.comp_cost * base_fitness
        else:
            competition_penalty = 0.0
        
        total_fitness = base_fitness + symbiotic_benefit - competition_penalty
        return max(0.1, total_fitness)  # Minimum survival fitness
    
    def evolve_generation(self) -> Dict:
        """
        Evolve one generation through symbiotic selection.
        """
        if not self.organisms:
            return {"error": "No organisms to evolve"}
        
        # Calculate fitness for all organisms
        fitness_scores = [self.calculate_fitness(org) for org in self.organisms]
        
        # Track symbiotic vs competitive organisms
        symbiotic_organisms = [org for org in self.organisms if len(org.symbiotic_partners) > 0]
        competitive_organisms = [org for org in self.organisms if len(org.symbiotic_partners) == 0]
        
        # Selection based on fitness
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return {"error": "Zero total fitness"}
        
        selection_probabilities = [f / total_fitness for f in fitness_scores]
        
        # Create next generation
        next_generation = []
        population_size = len(self.organisms)
        
        for i in range(population_size):
            # Select parent based on fitness
            parent_idx = np.random.choice(len(self.organisms), p=selection_probabilities)
            parent = self.organisms[parent_idx]
            
            # Create offspring with mutations
            offspring = Organism(
                id=len(next_generation) + len(self.organisms),
                coherence=max(0.1, parent.coherence + np.random.normal(0, 0.05)),
                fertility=max(0.1, parent.fertility + np.random.normal(0, 0.05)),
                metabolic_pathways=parent.metabolic_pathways.copy(),
                symbiotic_partners=[],
                generation=parent.generation + 1
            )
            
            # Pathway mutations
            if np.random.random() < 0.1:  # 10% chance of pathway mutation
                available_pathways = ['glycolysis', 'respiration', 'photosynthesis', 
                                    'nitrogen_fixation', 'sulfur_cycle', 'methane_production']
                new_pathway = np.random.choice(available_pathways)
                if new_pathway not in offspring.metabolic_pathways:
                    offspring.metabolic_pathways.append(new_pathway)
            
            next_generation.append(offspring)
        
        # Update organism population
        self.organisms = next_generation
        
        # Form new symbiotic relationships
        self.form_new_symbioses()
        
        # Record statistics
        stats = {
            'generation': self.organisms[0].generation,
            'total_organisms': len(self.organisms),
            'symbiotic_count': len([org for org in self.organisms if len(org.symbiotic_partners) > 0]),
            'competitive_count': len([org for org in self.organisms if len(org.symbiotic_partners) == 0]),
            'avg_coherence': np.mean([org.coherence for org in self.organisms]),
            'avg_fertility': np.mean([org.fertility for org in self.organisms]),
            'avg_fitness': np.mean([self.calculate_fitness(org) for org in self.organisms]),
            'avg_symbiotic_fitness': np.mean([self.calculate_fitness(org) for org in self.organisms if len(org.symbiotic_partners) > 0]) if symbiotic_organisms else 0,
            'avg_competitive_fitness': np.mean([self.calculate_fitness(org) for org in self.organisms if len(org.symbiotic_partners) == 0]) if competitive_organisms else 0
        }
        
        return stats
    
    def form_new_symbioses(self):
        """Attempt to form new symbiotic relationships in current population."""
        for i, org1 in enumerate(self.organisms):
            for j, org2 in enumerate(self.organisms[i+1:], i+1):
                if org2.id not in org1.symbiotic_partners:
                    compatibility = self.calculate_compatibility(org1, org2)
                    if compatibility > self.symb_threshold:
                        # Form symbiosis with some probability
                        if np.random.random() < compatibility:
                            self.form_symbiosis(org1, org2)
    
    def run_simulation(self, 
                      initial_population: List[Organism], 
                      generations: int = 100) -> Dict:
        """
        Run complete symbiotic evolution simulation.
        """
        self.organisms = initial_population.copy()
        
        history = {
            'generations': [],
            'symbiotic_counts': [],
            'competitive_counts': [],
            'avg_coherence': [],
            'avg_fertility': [],
            'avg_fitness': [],
            'symbiotic_advantage': []  # Fitness advantage of symbiotic organisms
        }
        
        for gen in range(generations):
            stats = self.evolve_generation()
            
            if 'error' in stats:
                break
            
            history['generations'].append(stats['generation'])
            history['symbiotic_counts'].append(stats['symbiotic_count'])
            history['competitive_counts'].append(stats['competitive_count'])
            history['avg_coherence'].append(stats['avg_coherence'])
            history['avg_fertility'].append(stats['avg_fertility'])
            history['avg_fitness'].append(stats['avg_fitness'])
            
            # Calculate symbiotic advantage
            if stats['avg_symbiotic_fitness'] > 0 and stats['avg_competitive_fitness'] > 0:
                advantage = stats['avg_symbiotic_fitness'] / stats['avg_competitive_fitness']
            else:
                advantage = 1.0
            history['symbiotic_advantage'].append(advantage)
        
        return {
            'simulation_history': history,
            'final_population': self.organisms,
            'total_generations': len(history['generations'])
        }


# examples/symbiotic_dynamics_demo.py
"""
Demonstration of Symbiotic Evolution Dynamics
Shows how SFH predicts symbiosis drives evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from src.sfh.dynamics.symbiotic_evolution import SymbioticEvolution, Organism

def create_initial_population(size: int = 50) -> List[Organism]:
    """Create diverse initial population for simulation."""
    organisms = []
    pathway_options = ['glycolysis', 'respiration', 'photosynthesis', 
                      'nitrogen_fixation', 'sulfur_cycle']
    
    for i in range(size):
        # Random initial traits
        coherence = np.random.uniform(0.3, 0.8)
        fertility = np.random.uniform(0.3, 0.8)
        
        # Random metabolic pathways (1-3 per organism)
        n_pathways = np.random.randint(1, 4)
        pathways = np.random.choice(pathway_options, n_pathways, replace=False).tolist()
        
        organism = Organism(
            id=i,
            coherence=coherence,
            fertility=fertility,
            metabolic_pathways=pathways,
            symbiotic_partners=[],
            generation=0
        )
        organisms.append(organism)
    
    return organisms

def run_demo():
    """Run symbiotic evolution demonstration."""
    print("SFH Symbiotic Evolution Demonstration")
    print("=" * 50)
    
    # Create simulation
    sim = SymbioticEvolution(
        cooperation_benefit=1.8,  # Strong benefit from cooperation
        competition_cost=0.4,     # Moderate cost of competition
        symbiosis_threshold=0.6   # Moderate threshold for symbiosis
    )
    
    # Initialize population
    initial_pop = create_initial_population(30)
    print(f"Initial population: {len(initial_pop)} organisms")
    
    # Run simulation
    results = sim.run_simulation(initial_pop, generations=50)
    
    # Analyze results
    history = results['simulation_history']
    print(f"\nSimulation completed: {results['total_generations']} generations")
    
    final_symbiotic = history['symbiotic_counts'][-1]
    final_competitive = history['competitive_counts'][-1]
    final_advantage = history['symbiotic_advantage'][-1]
    
    print(f"Final symbiotic organisms: {final_symbiotic}")
    print(f"Final competitive organisms: {final_competitive}")
    print(f"Symbiotic fitness advantage: {final_advantage:.2f}x")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    generations = history['generations']
    
    # Population composition
    ax1.plot(generations, history['symbiotic_counts'], 'b-', label='Symbiotic', linewidth=2)
    ax1.plot(generations, history['competitive_counts'], 'r-', label='Competitive', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Population Count')
    ax1.set_title('Population Composition Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average traits
    ax2.plot(generations, history['avg_coherence'], 'g-', label='Coherence', linewidth=2)
    ax2.plot(generations, history['avg_fertility'], 'm-', label='Fertility', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Trait Value')
    ax2.set_title('Evolution of Average Traits')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Fitness evolution
    ax3.plot(generations, history['avg_fitness'], 'k-', label='Average Fitness', linewidth=2)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Average Fitness')
    ax3.set_title('Fitness Evolution Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Symbiotic advantage
    ax4.plot(generations, history['symbiotic_advantage'], 'orange', linewidth=2)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No Advantage')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Fitness Ratio (Symbiotic/Competitive)')
    ax4.set_title('Symbiotic Fitness Advantage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('symbiotic_evolution_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    results = run_demo()


# examples/basic_coherence_fertility.py
"""
Basic Coherence-Fertility Optimization Example
Demonstrates core SFH mathematical framework from Chapter 4
"""

import numpy as np
import matplotlib.pyplot as plt
from src.sfh.core.optimization import SFHOptimizer
from src.sfh.core.coherence_model import CoherenceModel
from src.sfh.core.fertility_model import FertilityModel

def create_sample_system(n_nodes: int = 10) -> dict:
    """Create a sample biological system for optimization."""
    # Create random connectivity matrix (representing metabolic/regulatory networks)
    connectivity = np.random.rand(n_nodes, n_nodes) * 0.3
    # Make it symmetric for undirected connections
    connectivity = (connectivity + connectivity.T) / 2
    np.fill_diagonal(connectivity, 0)  # No self-connections
    
    return {
        'connectivity_matrix': connectivity,
        'n_nodes': n_nodes
    }

def demonstrate_optimization():
    """Demonstrate coherence-fertility optimization."""
    print("SFH Coherence-Fertility Optimization Demonstration")
    print("=" * 55)
    
    # Create test system
    system = create_sample_system(8)
    print(f"Test system: {system['n_nodes']} nodes")
    
    # Initialize optimizer with equal weights
    optimizer = SFHOptimizer(coherence_weight=0.5, fertility_weight=0.5)
    
    # Random initial parameters (stability + innovation vectors)
    n_params = system['n_nodes'] * 2  # stability and innovation for each node
    initial_params = np.random.rand(n_params) * 0.5
    
    print(f"Initial parameters: {n_params} dimensions")
    
    # Optimize system
    result = optimizer.optimize_system(initial_params, system)
    
    if result['success']:
        print("\nOptimization successful!")
        print(f"Iterations: {result['iterations']}")
        print(f"Function evaluations: {result['function_evaluations']}")
        print(f"Final coherence: {result['final_coherence']:.4f}")
        print(f"Final fertility: {result['final_fertility']:.4f}")
        print(f"Combined fitness: {result['combined_fitness']:.4f}")
    else:
        print(f"Optimization failed: {result['optimization_message']}")
        return
    
    # Compare different weight combinations
    weight_combinations = [
        (0.8, 0.2),  # Coherence-focused
        (0.5, 0.5),  # Balanced
        (0.2, 0.8),  # Fertility-focused
    ]
    
    results_comparison = []
    
    for w_c, w_f in weight_combinations:
        opt = SFHOptimizer(coherence_weight=w_c, fertility_weight=w_f)
        res = opt.optimize_system(initial_params, system)
        results_comparison.append({
            'weights': (w_c, w_f),
            'coherence': res['final_coherence'],
            'fertility': res['final_fertility'],
            'fitness': res['combined_fitness']
        })
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Weight comparison
    labels = ['Coherence-focused\n(0.8, 0.2)', 'Balanced\n(0.5, 0.5)', 'Fertility-focused\n(0.2, 0.8)']
    coherences = [r['coherence'] for r in results_comparison]
    fertilities = [r['fertility'] for r in results_comparison]
    fitnesses = [r['fitness'] for r in results_comparison]
    
    x = np.arange(len(labels))
    width = 0.25
    
    ax1.bar(x - width, coherences, width, label='Coherence', alpha=0.8, color='blue')
    ax1.bar(x, fertilities, width, label='Fertility', alpha=0.8, color='green')
    ax1.bar(x + width, fitnesses, width, label='Combined Fitness', alpha=0.8, color='red')
    
    ax1.set_xlabel('Optimization Strategy')
    ax1.set_ylabel('Value')
    ax1.set_title('Optimization Results by Weight Strategy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Evolutionary trajectory
    trajectory_result = optimizer.evolutionary_trajectory(
        initial_state={
            'parameters': initial_params,
            'system_data': system
        },
        n_generations=50,
        mutation_rate=0.02
    )
    
    generations = range(trajectory_result['generations'])
    ax2.plot(generations, trajectory_result['coherence_trajectory'], 'b-', 
             label='Coherence', linewidth=2, alpha=0.8)
    ax2.plot(generations, trajectory_result['fertility_trajectory'], 'g-', 
             label='Fertility', linewidth=2, alpha=0.8)
    ax2.plot(generations, trajectory_result['fitness_trajectory'], 'r-', 
             label='Combined Fitness', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Value')
    ax2.set_title('Evolutionary Trajectory (50 generations)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coherence_fertility_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nEvolutionary trajectory completed:")
    print(f"Final generation coherence: {trajectory_result['coherence_trajectory'][-1]:.4f}")
    print(f"Final generation fertility: {trajectory_result['fertility_trajectory'][-1]:.4f}")
    print(f"Final generation fitness: {trajectory_result['fitness_trajectory'][-1]:.4f}")
    
    return results_comparison, trajectory_result

if __name__ == "__main__":
    results = demonstrate_optimization()


# examples/mitochondrial_evolution.py
"""
Mitochondrial Evolution Simulation
Demonstrates SFH explanation of endosymbiotic theory from Chapter 6
"""

import numpy as np
import matplotlib.pyplot as plt
from src.sfh.dynamics.symbiotic_evolution import SymbioticEvolution, Organism

def create_prokaryotic_population() -> List[Organism]:
    """Create initial prokaryotic population before endosymbiosis."""
    organisms = []
    
    # Alpha-proteobacteria-like organisms (future mitochondria)
    for i in range(15):
        alpha_proto = Organism(
            id=i,
            coherence=np.random.uniform(0.6, 0.8),  # High metabolic coherence
            fertility=np.random.uniform(0.4, 0.6),   # Moderate reproduction
            metabolic_pathways=['respiration', 'electron_transport', 'atp_synthesis'],
            symbiotic_partners=[],
            generation=0
        )
        organisms.append(alpha_proto)
    
    # Early eukaryotic-like cells (hosts)
    for i in range(15, 25):
        host_cell = Organism(
            id=i,
            coherence=np.random.uniform(0.4, 0.6),   # Lower initial coherence
            fertility=np.random.uniform(0.6, 0.8),   # Higher reproduction potential
            metabolic_pathways=['glycolysis', 'fermentation'],
            symbiotic_partners=[],
            generation=0
        )
        organisms.append(host_cell)
    
    return organisms

def simulate_endosymbiosis():
    """Simulate the endosymbiotic origin of mitochondria."""
    print("Mitochondrial Endosymbiosis Simulation")
    print("=" * 40)
    
    # Create simulation with high cooperation benefits
    sim = SymbioticEvolution(
        cooperation_benefit=2.5,  # Very high benefit from endosymbiosis
        competition_cost=0.6,     # High cost of remaining separate
        symbiosis_threshold=0.4   # Lower threshold - easier symbiosis formation
    )
    
    # Initialize prokaryotic population
    initial_pop = create_prokaryotic_population()
    print(f"Initial population: {len(initial_pop)} organisms")
    print("- Alpha-proteobacteria-like: 15 organisms")
    print("- Early eukaryotic hosts: 10 organisms")
    
    # Run simulation
    results = sim.run_simulation(initial_pop, generations=100)
    
    # Analyze endosymbiotic relationships
    final_pop = results['final_population']
    
    # Identify successful endosymbiotic cells
    endosymbiotic_cells = []
    for org in final_pop:
        if len(org.symbiotic_partners) > 0:
            # Check if it has both respiratory and glycolytic pathways
            pathways = set(org.metabolic_pathways)
            if ('respiration' in pathways or 'electron_transport' in pathways) and 'glycolysis' in pathways:
                endosymbiotic_cells.append(org)
    
    print(f"\nFinal population: {len(final_pop)} organisms")
    print(f"Endosymbiotic cells (proto-eukaryotes): {len(endosymbiotic_cells)}")
    
    history = results['simulation_history']
    final_advantage = history['symbiotic_advantage'][-1] if history['symbiotic_advantage'] else 1.0
    print(f"Endosymbiotic fitness advantage: {final_advantage:.2f}x")
    
    # Calculate metabolic efficiency improvement
    if endosymbiotic_cells:
        avg_endo_coherence = np.mean([cell.coherence for cell in endosymbiotic_cells])
        avg_endo_fertility = np.mean([cell.fertility for cell in endosymbiotic_cells])
        
        # Compare to non-symbiotic organisms
        non_symbiotic = [org for org in final_pop if len(org.symbiotic_partners) == 0]
        if non_symbiotic:
            avg_non_coherence = np.mean([cell.coherence for cell in non_symbiotic])
            avg_non_fertility = np.mean([cell.fertility for cell in non_symbiotic])
            
            coherence_improvement = avg_endo_coherence / avg_non_coherence
            fertility_improvement = avg_endo_fertility / avg_non_fertility
            
            print(f"Coherence improvement: {coherence_improvement:.2f}x")
            print(f"Fertility improvement: {fertility_improvement:.2f}x")
    
    # Plot evolutionary trajectory
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    generations = history['generations']
    
    # Symbiosis formation over time
    ax1.plot(generations, history['symbiotic_counts'], 'purple', linewidth=3, 
             label='Endosymbiotic Cells')
    ax1.plot(generations, history['competitive_counts'], 'gray', linewidth=2, 
             label='Free-living Cells', linestyle='--')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Cell Count')
    ax1.set_title('Evolution of Endosymbiotic Relationships')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metabolic evolution
    ax2.plot(generations, history['avg_coherence'], 'blue', linewidth=2, 
             label='Metabolic Coherence')
    ax2.plot(generations, history['avg_fertility'], 'green', linewidth=2, 
             label='Reproductive Fertility')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Trait Value')
    ax2.set_title('Metabolic and Reproductive Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Fitness advantage of endosymbiosis
    ax3.plot(generations, history['symbiotic_advantage'], 'red', linewidth=3)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='No Advantage')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Fitness Ratio (Endosymbiotic/Free-living)')
    ax3.set_title('Endosymbiotic Fitness Advantage')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Final population analysis
    if endosymbiotic_cells and len(final_pop) > len(endosymbiotic_cells):
        endo_fitness = [sim.calculate_fitness(cell) for cell in endosymbiotic_cells]
        free_fitness = [sim.calculate_fitness(cell) for cell in final_pop 
                       if len(cell.symbiotic_partners) == 0]
        
        ax4.hist(endo_fitness, bins=10, alpha=0.7, label='Endosymbiotic Cells', 
                color='purple', density=True)
        if free_fitness:
            ax4.hist(free_fitness, bins=10, alpha=0.7, label='Free-living Cells', 
                    color='gray', density=True)
        ax4.set_xlabel('Fitness')
        ax4.set_ylabel('Density')
        ax4.set_title('Final Population Fitness Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for\nfitness distribution', 
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('mitochondrial_evolution_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, endosymbiotic_cells

if __name__ == "__main__":
    results, endo_cells = simulate_endosymbiosis()
