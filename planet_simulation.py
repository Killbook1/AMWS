import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
from sklearn.cluster import KMeans
import seaborn as sns
from collections import defaultdict
import networkx as nx
import dill

class HyperAdvancedPlanetSimulation:
    def __init__(self, size=300, time_acceleration=10000):
        self.size = size
        self.time_acceleration = time_acceleration
        self.year = 0
        
        # Environment
        self.elevation = self.generate_terrain()
        self.temperature = self.initialize_temperature()
        self.rainfall = np.zeros((size, size))
        self.wind = np.zeros((size, size, 2))  # 2D vector field
        self.ocean_currents = np.zeros((size, size, 2))
        
        # Resources
        self.resources = {
            'water': np.zeros((size, size)),
            'vegetation': np.zeros((size, size)),
            'minerals': np.random.power(0.5, (size, size)),
            'oxygen': np.full((size, size), 0.21),  # Earth-like initial oxygen level
            'carbon_dioxide': np.full((size, size), 0.0004)  # Earth-like initial CO2 level
        }
        
        # Species
        self.species = defaultdict(lambda: {'population': np.zeros((size, size)), 'genome': {}})
        self.food_web = nx.DiGraph()
        
        # Civilization
        self.technology_level = 0
        self.pollution = np.zeros((size, size))
        self.cities = []
        
        # Geological activity
        self.tectonic_plates = self.initialize_tectonic_plates()
        self.volcanic_activity = np.zeros((size, size))
        
        self.initialize_simulation()

    def generate_terrain(self):
        noise = np.random.rand(self.size, self.size)
        kernel = np.outer(np.hanning(40), np.hanning(40))
        terrain = convolve(noise, kernel, mode='wrap')
        return (terrain - terrain.min()) / (terrain.max() - terrain.min())

    def initialize_temperature(self):
        base_temp = np.linspace(40, -40, self.size).reshape(-1, 1)
        return base_temp + 5 * self.elevation + np.random.normal(0, 2, (self.size, self.size))

    def initialize_tectonic_plates(self):
        kmeans = KMeans(n_clusters=7, random_state=42).fit(self.elevation.reshape(-1, 1))
        return kmeans.labels_.reshape(self.size, self.size)

    def initialize_simulation(self):
        self.update_climate()
        self.initialize_resources()
        self.initialize_species()

    def initialize_resources(self):
        self.resources['water'] = 0.5 * (1 - self.elevation) + 0.5 * self.rainfall
        self.resources['vegetation'] = 0.3 * self.resources['water'] * (1 - 0.1 * np.abs(self.temperature))

    def initialize_species(self):
        habitable = (self.temperature > 0) & (self.temperature < 30) & (self.resources['water'] > 0.2)
        
        for species_type in ['plants', 'herbivores', 'carnivores', 'decomposers']:
            self.species[species_type]['population'][habitable] = np.random.exponential(0.1, habitable.sum())
            self.species[species_type]['genome'] = self.generate_genome()
        
        self.species['humans']['population'][habitable] = np.random.exponential(0.01, habitable.sum())
        self.species['humans']['genome'] = self.generate_genome(intelligence=0.5)
        
        self.update_food_web()

    def generate_genome(self, intelligence=0.1):
        return {
            'size': np.random.normal(0.5, 0.1),
            'speed': np.random.normal(0.5, 0.1),
            'intelligence': np.random.normal(intelligence, 0.1),
            'temperature_tolerance': np.random.normal(0.5, 0.1),
            'water_need': np.random.normal(0.5, 0.1),
            'lifespan': np.random.normal(0.5, 0.1),
            'mutation_rate': np.random.normal(0.01, 0.001)
        }

    def update_food_web(self):
        self.food_web.clear()
        self.food_web.add_edge('plants', 'herbivores')
        self.food_web.add_edge('herbivores', 'carnivores')
        self.food_web.add_edge('plants', 'decomposers')
        self.food_web.add_edge('herbivores', 'decomposers')
        self.food_web.add_edge('carnivores', 'decomposers')

    def update_climate(self):
        # Update temperature
        self.temperature += np.random.normal(0, 0.1, (self.size, self.size))
        self.temperature += 0.01 * (15 - self.temperature)  # Regress to mean
        
        # Update wind
        self.wind += np.random.normal(0, 0.1, (self.size, self.size, 2))
        self.wind *= 0.9  # Damping
        
        # Update ocean currents
        self.ocean_currents += 0.1 * self.wind
        self.ocean_currents *= 0.95  # Damping
        
        # Update rainfall
        moisture = 0.5 * self.resources['water']
        self.rainfall = convolve(moisture, np.ones((5, 5))/25, mode='wrap') + 0.1 * np.linalg.norm(self.wind, axis=2)
        self.rainfall = np.clip(self.rainfall, 0, 1)

    def update_resources(self):
        # Water dynamics
        self.resources['water'] += 0.8 * self.rainfall - 0.2 * self.temperature / 40
        self.resources['water'] = np.clip(self.resources['water'], 0, 1)

        # Vegetation growth
        growth_rate = 0.05 * (1 + np.sin(self.year / 10))  # Seasonal variation
        self.resources['vegetation'] += growth_rate * self.resources['water'] * (1 - self.resources['vegetation'])
        self.resources['vegetation'] -= 0.01 * self.species['herbivores']['population'] + 0.005 * self.species['humans']['population']
        self.resources['vegetation'] = np.clip(self.resources['vegetation'], 0, 1)

        # Mineral depletion
        self.resources['minerals'] -= 0.001 * self.species['humans']['population'] * (1 + self.technology_level / 100)
        self.resources['minerals'] = np.clip(self.resources['minerals'], 0, 1)

        # Oxygen and CO2 dynamics
        self.resources['oxygen'] += 0.001 * self.resources['vegetation'] - 0.0005 * sum(self.species[s]['population'] for s in ['herbivores', 'carnivores', 'humans'])
        self.resources['carbon_dioxide'] -= 0.001 * self.resources['vegetation'] + 0.0005 * sum(self.species[s]['population'] for s in ['herbivores', 'carnivores', 'humans'])
        self.resources['oxygen'] = np.clip(self.resources['oxygen'], 0, 0.3)
        self.resources['carbon_dioxide'] = np.clip(self.resources['carbon_dioxide'], 0, 0.001)

    def update_species(self):
        for species, data in self.species.items():
            population = data['population']
            genome = data['genome']

            if species == 'plants':
                food = self.resources['water'] * self.resources['carbon_dioxide']
                predation = 0.1 * self.species['herbivores']['population']
            elif species == 'herbivores':
                food = self.resources['vegetation']
                predation = 0.1 * self.species['carnivores']['population']
            elif species == 'carnivores':
                food = 0.1 * self.species['herbivores']['population']
                predation = 0.01 * self.species['humans']['population'] * (1 + self.technology_level / 100)
            elif species == 'decomposers':
                food = 0.01 * sum(self.species[s]['population'] for s in ['plants', 'herbivores', 'carnivores'])
                predation = 0
            else:  # humans
                food = 0.1 * self.resources['vegetation'] + 0.05 * self.species['herbivores']['population'] + 0.02 * self.species['carnivores']['population']
                food *= (1 + self.technology_level / 100)
                predation = 0

            birth_rate = 0.2 * food * genome['lifespan']
            death_rate = 0.1 * (1 - food) + 0.05 * np.abs(self.temperature - 20) / 30 + predation + 0.01 * self.pollution
            death_rate *= (2 - genome['temperature_tolerance'])
            population += (birth_rate - death_rate) * population
            population = np.clip(population, 0, 100)

            # Migration
            population = gaussian_filter(population, sigma=genome['speed'])

            # Evolution
            if np.random.random() < genome['mutation_rate']:
                for trait in genome:
                    if trait != 'mutation_rate':
                        genome[trait] += np.random.normal(0, 0.01)
                        genome[trait] = np.clip(genome[trait], 0, 1)

            self.species[species]['population'] = population
            self.species[species]['genome'] = genome

    def update_civilization(self):
        human_pop = self.species['humans']['population'].sum()
        if human_pop > 0:
            self.technology_level += 0.01 * np.log(human_pop) * self.species['humans']['genome']['intelligence']
            self.pollution += 0.001 * self.species['humans']['population'] * (1 + self.technology_level / 50)
            self.pollution = np.clip(self.pollution, 0, 1)

            # City formation
            if self.year % 10 == 0:  # Check for new cities every 10 years
                potential_city = np.random.rand(self.size, self.size) < self.species['humans']['population'] / 100
                new_cities = np.argwhere(potential_city & (self.elevation > 0.1))
                for city in new_cities:
                    if len(self.cities) < 50 and all(np.linalg.norm(city - c) > 10 for c in self.cities):
                        self.cities.append(city)

    def update_geology(self):
        # Plate movement
        self.tectonic_plates = np.roll(self.tectonic_plates, (1, 1), (0, 1))
        
        # Volcanic activity
        plate_boundaries = convolve(self.tectonic_plates, np.ones((3, 3)), mode='wrap') != 9
        self.volcanic_activity = np.random.rand(*plate_boundaries.shape) * plate_boundaries
        
        # Update elevation based on volcanic activity
        self.elevation += 0.001 * self.volcanic_activity
        self.elevation = np.clip(self.elevation, 0, 1)

    def natural_disaster(self):
        if np.random.random() < 0.05:  # 5% chance each year
            disaster_type = np.random.choice(['earthquake', 'hurricane', 'volcanic_eruption', 'meteor_impact'])
            impact_area = np.random.rand(self.size, self.size) < 0.2  # 20% of the planet affected

            if disaster_type == 'earthquake':
                self.elevation[impact_area] += np.random.normal(0, 0.1, impact_area.sum())
            elif disaster_type == 'hurricane':
                self.resources['water'][impact_area] += 0.5
                for species in self.species.values():
                    species['population'][impact_area] *= 0.9
            elif disaster_type == 'volcanic_eruption':
                self.temperature[impact_area] += 10
                self.resources['vegetation'][impact_area] *= 0.5
                self.resources['minerals'][impact_area] += 0.2
            else:  # meteor_impact
                crater = np.random.randint(0, self.size, 2)
                crater_size = np.random.randint(5, 20)
                y, x = np.ogrid[-crater[0]:self.size-crater[0], -crater[1]:self.size-crater[1]]
                mask = x*x + y*y <= crater_size*crater_size
                self.elevation[mask] -= 0.5
                for species in self.species.values():
                    species['population'][mask] *= 0.1

            print(f"Year {self.year}: {disaster_type.capitalize()} occurred!")

    def simulate_year(self):
        self.update_climate()
        self.update_resources()
        self.update_species()
        self.update_civilization()
        self.update_geology()
        self.natural_disaster()
        self.year += 1

    def run_simulation(self, years):
        target_year = self.year + years
        while self.year < target_year:
              self.simulate_year()
              if self.year % 100 == 0:
                 print(f"Year {self.year}")
                 self.analyze_data()

    def analyze_data(self):
        total_pop = sum(data['population'].sum() for data in self.species.values())
        biodiversity = len([sp for sp in self.species.values() if sp['population'].sum() > 0])
        avg_temp = self.temperature.mean()
        resource_abundance = {res: val.mean() for res, val in self.resources.items()}
        pollution_level = self.pollution.mean()
        
        print(f"Year {self.year} Analysis:")
        print(f"Total Population: {total_pop:.2f}")
        print(f"Biodiversity: {biodiversity}")
        print(f"Average Temperature: {avg_temp:.2f}°C")
        print(f"Resource Abundance: {resource_abundance}")
        print(f"Technology Level: {self.technology_level:.2f}")
        print(f"Pollution Level: {pollution_level:.2f}")
        print(f"Number of Cities: {len(self.cities)}")
        
        for species, data in self.species.items():
            print(f"{species.capitalize()} Genome:")
            for trait, value in data['genome'].items():
                print(f"  {trait}: {value:.2f}")

    def visualize(self):
       fig, axs = plt.subplots(3, 4, figsize=(20, 20))

       # Plot elevation
       ax1 = axs[0, 0]
       x, y = np.meshgrid(range(self.size), range(self.size))
       elevation_plot = ax1.imshow(self.elevation, cmap='terrain', extent=[0, self.size, 0, self.size])
       fig.colorbar(elevation_plot, ax=ax1, label='Elevation')
       ax1.set_title('Elevation and Tectonic Plates')
       ax1.set_xlabel('X')
       ax1.set_ylabel('Y')

       # Plot temperature
       ax2 = axs[0, 1]
       temperature_plot = ax2.imshow(self.temperature, cmap='coolwarm', extent=[0, self.size, 0, self.size])
       fig.colorbar(temperature_plot, ax=ax2, label='Temperature (°C)')
       ax2.set_title('Temperature')
       ax2.set_xlabel('X')
       ax2.set_ylabel('Y')

       # Plot rainfall and wind
       ax3 = axs[0, 2]
       rainfall_plot = ax3.imshow(self.rainfall, cmap='Blues', extent=[0, self.size, 0, self.size])
       wind_skip = (slice(None, None, 10), slice(None, None, 10))
       ax3.quiver(x[wind_skip], y[wind_skip], self.wind[wind_skip][..., 0], self.wind[wind_skip][..., 1], scale=50)
       fig.colorbar(rainfall_plot, ax=ax3, label='Rainfall')
       ax3.set_title('Rainfall and Wind')
       ax3.set_xlabel('X')
       ax3.set_ylabel('Y')

       # Plot resources
       resource_names = ['water', 'vegetation', 'minerals', 'oxygen', 'carbon_dioxide']
       for i, res in enumerate(resource_names):
            ax = axs[1, i]
            resource_plot = ax.imshow(self.resources[res], cmap='YlGn', extent=[0, self.size, 0, self.size])
            fig.colorbar(resource_plot, ax=ax, label=f'{res.capitalize()}')
            ax.set_title(f'{res.capitalize()} Distribution')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

       # Plot species populations
       ax9 = axs[2, 0]
       species_total = {species: data['population'].sum() for species, data in self.species.items()}
       ax9.bar(species_total.keys(), species_total.values())
       ax9.set_title('Species Population')
       ax9.set_xlabel('Species')
       ax9.set_ylabel('Population')
       ax9.set_yscale('log')

       # Plot food web
       ax10 = axs[2, 1]
       pos = nx.spring_layout(self.food_web)
       nx.draw_networkx_nodes(self.food_web, pos, ax=ax10, node_color='lightblue', node_size=3000)
       nx.draw_networkx_labels(self.food_web, pos, ax=ax10, font_size=8, font_weight='bold')
       nx.draw_networkx_edges(self.food_web, pos, ax=ax10, edge_color='gray')
       ax10.set_title('Food Web')

       # Plot civilization
       ax11 = axs[2, 2]
       pollution_plot = ax11.imshow(self.pollution, cmap='Reds', extent=[0, self.size, 0, self.size])
       fig.colorbar(pollution_plot, ax=ax11, label='Pollution')
       for city in self.cities:
           ax11.plot(city[1], city[0], 'ko', markersize=5)
       ax11.set_title('Pollution and Cities')
       ax11.set_xlabel('X')
       ax11.set_ylabel('Y')

       # Plot geological activity
       ax12 = axs[2, 3]
       volcanic_activity_plot = ax12.imshow(self.volcanic_activity, cmap='magma', extent=[0, self.size, 0, self.size])
       fig.colorbar(volcanic_activity_plot, ax=ax12, label='Volcanic Activity')
       ax12.set_title('Volcanic Activity')
       ax12.set_xlabel('X')
       ax12.set_ylabel('Y')

       plt.tight_layout()
       plt.show()

    def run_advanced_analysis(self):
        # Species diversity analysis
        diversity_index = {}
        for species, data in self.species.items():
            population = data['population']
            total = population.sum()
            if total > 0:
               p = population / total
               diversity_index[species] = -np.sum(p * np.log(p + 1e-10))  # Shannon diversity index

        # Climate change analysis
        temp_change = np.mean(self.temperature) - np.mean(self.initialize_temperature())
        rainfall_change = np.mean(self.rainfall) - 0.5  # Assuming 0.5 was the initial average rainfall

        # Resource depletion analysis
        resource_depletion = {res: 1 - np.mean(val) for res, val in self.resources.items()}

        # Civilization impact analysis
        human_impact = np.mean(self.pollution) * self.technology_level

        # Ecological balance analysis
        ecological_balance = np.std([data['population'].sum() for data in self.species.values()])

        print("Advanced Analysis Results:")
        print("Species Diversity (Shannon Index):")
        for species, index in diversity_index.items():
            print(f"  {species}: {index:.2f}")
            print(f"Temperature Change: {temp_change:.2f}°C")
            print(f"Rainfall Change: {rainfall_change:.2f}")
            print("Resource Depletion:")
        for res, depl in resource_depletion.items():
            print(f"  {res}: {depl:.2%}")
            print(f"Human Impact: {human_impact:.2f}")
            print(f"Ecological Balance (Population Standard Deviation): {ecological_balance:.2f}")

    def save_simulation_state(self, filename):
        with open(filename, 'wb') as f:
             dill.dump(self.__dict__, f)

    def load_simulation_state(self, filename):
        with open(filename, 'rb') as f:
             self.__dict__.update(dill.load(f))

if __name__ == "__main__":
    # Create the simulation instance
    sim = HyperAdvancedPlanetSimulation(size=1500, time_acceleration=100000)

    # Run the simulation
    print("Starting simulation...")
    sim.run_simulation(years=10000)
    print("Simulation complete.")

    # Perform advanced analysis
    print("Running advanced analysis...")
    sim.run_advanced_analysis()

    # Visualize the results
    print("Generating visualizations...")
    sim.visualize()

    # Save the simulation state
    print("Saving simulation state...")
    sim.save_simulation_state('planet_simulation_state.pkl')
    print("Simulation state saved. Process complete.")